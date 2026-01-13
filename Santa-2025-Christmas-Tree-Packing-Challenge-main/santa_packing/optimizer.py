"""JAX simulated annealing optimizers for the packing problem.

This module implements the core SA kernels used by:
- `santa_packing.main` (single-instance runner)
- `santa_packing.cli.generate_submission` (submission generation pipeline)

The focus is on fast, JIT-friendly operations and broad-phase collision
filtering (circle/AABB) with an exact polygon intersection check when needed.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp

from .collisions import check_any_collisions
from .geometry import transform_polygon
from .l2o import policy_apply
from .packing import (
    packing_score_from_bboxes,
    prefix_packing_score_from_bboxes,
)
from .physics import polygons_intersect
from .tree import get_tree_polygon
from .tree_bounds import TREE_AABB_TABLE, TREE_AABB_TABLE_PADDED, TREE_RADIUS2, aabb_for_poses, theta_to_aabb_bin


def sa_check_collisions(poses, base_poly):
    """Check if any pair of trees intersect (SA helper).

    Args:
        poses: Array `(n, 3)` containing `(x, y, deg)` for each tree.
        base_poly: Base tree polygon `(v, 2)` in local coordinates.

    Returns:
        `True` if any collision exists, `False` otherwise.
    """
    return check_any_collisions(poses, base_poly)


@partial(
    jax.jit,
    static_argnames=["n_steps", "n_trees", "objective", "cooling", "proposal", "allow_collisions", "teleport_tries"],
)
def run_sa_batch(
    random_key,
    n_steps,
    n_trees,
    initial_poses,
    t_start=1.0,
    t_end=0.001,
    trans_sigma=0.1,
    rot_sigma=15.0,
    rot_prob=0.3,
    rot_prob_end=-1.0,
    swap_prob=0.0,
    swap_prob_end=-1.0,
    cooling="geom",
    cooling_power=1.0,
    trans_sigma_nexp=0.0,
    rot_sigma_nexp=0.0,
    sigma_nref=50.0,
    proposal="random",
    smart_prob=1.0,
    smart_beta=8.0,
    smart_drift=1.0,
    smart_noise=0.25,
    push_prob=0.1,
    push_scale=1.0,
    push_square_prob=0.5,
    compact_prob=0.0,
    compact_prob_end=-1.0,
    compact_scale=1.0,
    compact_square_prob=0.75,
    teleport_prob=0.0,
    teleport_prob_end=-1.0,
    teleport_tries=4,
    teleport_anchor_beta=6.0,
    teleport_ring_mult=1.02,
    teleport_jitter=0.05,
    overlap_lambda=0.0,
    adapt_sigma=True,
    accept_target=0.35,
    adapt_alpha=0.05,
    adapt_rate=0.1,
    trans_sigma_min=1e-4,
    trans_sigma_max=5.0,
    rot_sigma_min=1e-3,
    rot_sigma_max=90.0,
    adapt_rot_prob=False,
    rot_prob_adapt_rate=0.02,
    rot_prob_pull=0.02,
    rot_prob_min=0.0,
    rot_prob_max=1.0,
    reheat_patience=200,
    reheat_factor=1.0,
    reheat_decay=0.99,
    allow_collisions=False,
    objective="packing",
):
    """
    Runs a batch of SA chains.

    Args:
        random_key: JAX PRNGKey
        n_steps: Number of SA steps
        n_trees: Number of trees per packing
        initial_poses: (Batch, N, 3) or None to init?

    Returns:
        Final poses (Batch, N, 3) and scores (Batch,)
    """

    base_poly = get_tree_polygon()
    score_from_bboxes_fn = prefix_packing_score_from_bboxes if objective == "prefix" else packing_score_from_bboxes

    # Cache AABB tables in the same dtype as poses (avoids repeated recompute/casts).
    table = TREE_AABB_TABLE.astype(initial_poses.dtype)
    table_padded = TREE_AABB_TABLE_PADDED.astype(initial_poses.dtype)

    # Spatial hashing parameters (broad-phase): cell ~ diameter of bounding circle.
    circle_margin = jnp.asarray(1.0 + 1e-4, dtype=initial_poses.dtype)
    thr2 = jnp.asarray(4.0, dtype=initial_poses.dtype) * TREE_RADIUS2.astype(initial_poses.dtype)
    thr2_coll = thr2 * circle_margin
    radius = jnp.sqrt(TREE_RADIUS2).astype(initial_poses.dtype)
    cell_size = (2.0 * radius) * circle_margin
    inv_cell = jnp.asarray(1.0, dtype=initial_poses.dtype) / (cell_size + jnp.asarray(1e-12, dtype=initial_poses.dtype))

    def _cells_for_xy(xy: jax.Array) -> jax.Array:
        return jnp.floor(xy * inv_cell).astype(jnp.int32)

    def _aabb_for_pose(pose: jax.Array, *, padded: bool) -> jax.Array:
        tab = table_padded if padded else table
        idx = theta_to_aabb_bin(pose[2])
        local = tab[idx]
        xy = pose[0:2]
        xyxy = jnp.concatenate([xy, xy], axis=0)
        return local + xyxy

    def _check_collision_for_index_cached(
        poses_one: jax.Array,
        bboxes_padded_one: jax.Array,
        cells_one: jax.Array,
        idx: jax.Array,
    ) -> jax.Array:
        n = poses_one.shape[0]
        not_self = jnp.arange(n) != idx

        cell_k = cells_one[idx]
        dx = jnp.abs(cells_one[:, 0] - cell_k[0]) <= 1
        dy = jnp.abs(cells_one[:, 1] - cell_k[1]) <= 1
        neighbor = dx & dy & not_self

        bbox_k = bboxes_padded_one[idx]
        bbox_overlap = (
            (bbox_k[2] >= bboxes_padded_one[:, 0])
            & (bboxes_padded_one[:, 2] >= bbox_k[0])
            & (bbox_k[3] >= bboxes_padded_one[:, 1])
            & (bboxes_padded_one[:, 3] >= bbox_k[1])
        )

        centers = poses_one[:, :2]
        center_k = centers[idx]
        d = centers - center_k
        dist2 = jnp.sum(d * d, axis=1)
        circle = dist2 <= thr2_coll

        candidate = neighbor & bbox_overlap & circle

        def _check_candidates() -> jax.Array:
            poly_k = transform_polygon(base_poly, poses_one[idx])

            def _check_one(pose_j: jax.Array, do_test: jax.Array) -> jax.Array:
                return jax.lax.cond(
                    do_test,
                    lambda: polygons_intersect(poly_k, transform_polygon(base_poly, pose_j)),
                    lambda: jnp.array(False),
                )

            hits = jax.vmap(_check_one)(poses_one, candidate)
            return jnp.any(hits)

        return jax.lax.cond(jnp.any(candidate), _check_candidates, lambda: jnp.array(False))

    n_ratio = jnp.asarray(float(n_trees), dtype=initial_poses.dtype) / jnp.asarray(
        sigma_nref, dtype=initial_poses.dtype
    )
    trans_sigma_eff = jnp.asarray(trans_sigma, dtype=initial_poses.dtype) * (
        n_ratio ** jnp.asarray(trans_sigma_nexp, dtype=initial_poses.dtype)
    )
    rot_sigma_eff = jnp.asarray(rot_sigma, dtype=initial_poses.dtype) * (
        n_ratio ** jnp.asarray(rot_sigma_nexp, dtype=initial_poses.dtype)
    )
    overlap_lambda_t = jnp.asarray(overlap_lambda, dtype=initial_poses.dtype)

    push_prob_t = jnp.asarray(push_prob, dtype=initial_poses.dtype)
    push_scale_t = jnp.asarray(push_scale, dtype=initial_poses.dtype)
    push_square_prob_t = jnp.asarray(push_square_prob, dtype=initial_poses.dtype)

    compact_prob_t = jnp.asarray(compact_prob, dtype=initial_poses.dtype)
    compact_scale_t = jnp.asarray(compact_scale, dtype=initial_poses.dtype)
    compact_square_prob_t = jnp.asarray(compact_square_prob, dtype=initial_poses.dtype)

    teleport_prob_t = jnp.asarray(teleport_prob, dtype=initial_poses.dtype)
    teleport_anchor_beta_t = jnp.asarray(teleport_anchor_beta, dtype=initial_poses.dtype)
    teleport_ring_mult_t = jnp.asarray(teleport_ring_mult, dtype=initial_poses.dtype)
    teleport_jitter_t = jnp.asarray(teleport_jitter, dtype=initial_poses.dtype)

    adapt_sigma_t = jnp.asarray(adapt_sigma, dtype=bool)
    accept_target_t = jnp.asarray(accept_target, dtype=initial_poses.dtype)
    adapt_alpha_t = jnp.asarray(adapt_alpha, dtype=initial_poses.dtype)
    adapt_rate_t = jnp.asarray(adapt_rate, dtype=initial_poses.dtype)
    trans_sigma_min_t = jnp.asarray(trans_sigma_min, dtype=initial_poses.dtype)
    trans_sigma_max_t = jnp.asarray(trans_sigma_max, dtype=initial_poses.dtype)
    rot_sigma_min_t = jnp.asarray(rot_sigma_min, dtype=initial_poses.dtype)
    rot_sigma_max_t = jnp.asarray(rot_sigma_max, dtype=initial_poses.dtype)

    adapt_rot_prob_t = jnp.asarray(adapt_rot_prob, dtype=bool)
    rot_prob_adapt_rate_t = jnp.asarray(rot_prob_adapt_rate, dtype=initial_poses.dtype)
    rot_prob_pull_t = jnp.asarray(rot_prob_pull, dtype=initial_poses.dtype)
    rot_prob_min_t = jnp.asarray(rot_prob_min, dtype=initial_poses.dtype)
    rot_prob_max_t = jnp.asarray(rot_prob_max, dtype=initial_poses.dtype)

    reheat_patience_t = jnp.asarray(reheat_patience, dtype=jnp.int32)
    reheat_factor_t = jnp.asarray(reheat_factor, dtype=initial_poses.dtype)
    reheat_decay_t = jnp.asarray(reheat_decay, dtype=initial_poses.dtype)
    reheat_enabled = reheat_patience_t > 0

    def _pair_penalty_for_index(poses_xy: jax.Array, idx: jax.Array) -> jax.Array:
        center_k = poses_xy[idx]
        d = poses_xy - center_k
        dist2 = jnp.sum(d * d, axis=1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        mask = (jnp.arange(n_trees) != idx).astype(pen.dtype)
        return jnp.sum((pen * pen) * mask)

    def _total_penalty(poses_xy: jax.Array) -> jax.Array:
        d = poses_xy[:, None, :] - poses_xy[None, :, :]
        dist2 = jnp.sum(d * d, axis=-1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        pen2 = pen * pen
        mask = jnp.triu(jnp.ones_like(pen2), k=1)
        return jnp.sum(pen2 * mask)

    def _select_bbox_inward(subkey: jax.Array, bboxes: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        min_x = jnp.min(bboxes[:, 0])
        min_y = jnp.min(bboxes[:, 1])
        max_x = jnp.max(bboxes[:, 2])
        max_y = jnp.max(bboxes[:, 3])
        width = max_x - min_x
        height = max_y - min_y
        use_x = width >= height

        slack_x = jnp.minimum(bboxes[:, 0] - min_x, max_x - bboxes[:, 2])
        slack_y = jnp.minimum(bboxes[:, 1] - min_y, max_y - bboxes[:, 3])
        slack = jnp.where(use_x, slack_x, slack_y)
        slack = jnp.maximum(slack, 0.0)
        scale = jnp.maximum(jnp.where(use_x, width, height), 1e-6)
        logits = -(slack / scale) * jnp.asarray(smart_beta, dtype=bboxes.dtype)
        idx = jax.random.categorical(subkey, logits)
        center = jnp.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=bboxes.dtype)
        return idx, center, use_x

    def step_fn(state, i):
        (
            key,
            poses,
            bboxes,
            bboxes_padded,
            cells,
            current_score,
            current_penalty,
            current_colliding,
            best_poses,
            best_score,
            trans_sigma_dyn,
            rot_sigma_dyn,
            rot_prob_dyn,
            acc_trans_ema,
            acc_rot_ema,
            stall_steps,
            reheat_level,
            temp,
        ) = state

        # Annealing schedule
        frac = i / n_steps
        anneal = frac ** jnp.asarray(cooling_power, dtype=initial_poses.dtype)
        if cooling in {"geom", "geometric", "exp", "exponential"}:
            current_temp = t_start * (t_end / t_start) ** anneal
        elif cooling == "linear":
            current_temp = t_start + (t_end - t_start) * anneal
        elif cooling == "log":
            # Log schedule with exact endpoints: T(0)=t_start, T(n_steps-1)=t_end.
            denom = jnp.log1p(jnp.asarray(float(max(n_steps - 1, 1)), dtype=initial_poses.dtype))
            alpha = (t_start / t_end - 1.0) / (denom + 1e-12)
            current_temp = t_start / (1.0 + alpha * jnp.log1p(i.astype(initial_poses.dtype)))
        else:
            current_temp = t_start * (t_end / t_start) ** anneal

        # Optional "reheating": scale the temperature up when we stall.
        temp_eff = current_temp * (jnp.asarray(1.0, dtype=initial_poses.dtype) + reheat_level)

        rot_prob_final = jnp.where(rot_prob_end >= 0.0, rot_prob_end, rot_prob)
        rot_prob_base = rot_prob + (rot_prob_final - rot_prob) * anneal
        rot_prob_base = jnp.clip(rot_prob_base, 0.0, 1.0)
        current_rot_prob = jnp.where(adapt_rot_prob_t, rot_prob_dyn, rot_prob_base)
        current_rot_prob = jnp.clip(current_rot_prob, rot_prob_min_t, rot_prob_max_t)

        swap_prob_final = jnp.where(swap_prob_end >= 0.0, swap_prob_end, swap_prob)
        current_swap_prob = swap_prob + (swap_prob_final - swap_prob) * anneal
        current_swap_prob = jnp.clip(current_swap_prob, 0.0, 1.0)

        # 1. Propose Move
        (
            key,
            subkey_select,
            subkey_k,
            subkey_k2,
            subkey_swap,
            subkey_rot_choice,
            subkey_trans,
            subkey_rot,
            subkey_gate,
            subkey_push,
            subkey_push_axis,
            subkey_compact,
            subkey_compact_axis,
            subkey_teleport,
            subkey_teleport_anchor,
            subkey_teleport_phi,
            subkey_teleport_noise,
        ) = jax.random.split(key, 17)

        batch_size = poses.shape[0]
        batch_idx = jnp.arange(batch_size)

        k_rand = jax.random.randint(subkey_k, (batch_size,), 0, n_trees)

        # Swap is a permutation move (useful for objective='prefix'); it does not change
        # the set of poses (only their order), so it cannot introduce new collisions.
        swap_choice = jax.random.uniform(subkey_swap, (batch_size,))
        if n_trees > 1:
            do_swap = swap_choice < current_swap_prob
        else:
            do_swap = jnp.zeros((batch_size,), dtype=bool)
        rot_choice = jax.random.uniform(subkey_rot_choice, (batch_size,))

        eps_xy = jax.random.normal(subkey_trans, (batch_size, 2))
        eps_theta = jax.random.normal(subkey_rot, (batch_size,))
        trans_scale = trans_sigma_dyn * temp_eff
        rot_scale = rot_sigma_dyn * temp_eff

        keys_pick = jax.random.split(subkey_select, batch_size)
        k_boundary, pack_center, use_x = jax.vmap(_select_bbox_inward)(keys_pick, bboxes)
        xy_k = poses[batch_idx, k_boundary, 0:2]
        direction = pack_center - xy_k
        axis_direction = jnp.where(
            use_x[:, None],
            jnp.stack([direction[:, 0], jnp.zeros_like(direction[:, 0])], axis=1),
            jnp.stack([jnp.zeros_like(direction[:, 1]), direction[:, 1]], axis=1),
        )
        norm = jnp.linalg.norm(axis_direction, axis=1, keepdims=True)
        unit_axis = axis_direction / (norm + 1e-12)
        drift = unit_axis * (trans_scale[:, None] * jnp.asarray(smart_drift, dtype=poses.dtype))
        noise = eps_xy * (trans_scale[:, None] * jnp.asarray(smart_noise, dtype=poses.dtype))
        dxy_smart = drift + noise

        dxy_rand = eps_xy * trans_scale[:, None]

        if proposal in {"bbox", "bbox_inward", "inward", "smart"}:
            k = k_boundary
            dxy = dxy_smart
        elif proposal == "mixed":
            gate = jax.random.uniform(subkey_gate, (batch_size,)) < jnp.asarray(smart_prob, dtype=poses.dtype)
            k = jnp.where(gate, k_boundary, k_rand)
            dxy = jnp.where(gate[:, None], dxy_smart, dxy_rand)
        else:
            k = k_rand
            dxy = dxy_rand

        dtheta = eps_theta * rot_scale

        rot_mask = rot_choice < current_rot_prob

        # Optional deterministic "push" move (translation-only): pick the tree
        # maximizing max(|x|,|y|) and nudge it toward the origin.
        push_choice = jax.random.uniform(subkey_push, (batch_size,))
        push_axis_choice = jax.random.uniform(subkey_push_axis, (batch_size,))
        abs_xy = jnp.abs(poses[:, :, 0:2])
        push_metric = jnp.maximum(abs_xy[:, :, 0], abs_xy[:, :, 1])
        k_push = jnp.argmax(push_metric, axis=1).astype(jnp.int32)
        xy_push = poses[batch_idx, k_push, 0:2]
        norm = jnp.linalg.norm(xy_push, axis=1, keepdims=True)
        unit_vec = -xy_push / (norm + 1e-12)
        axis_x = jnp.abs(xy_push[:, 0]) >= jnp.abs(xy_push[:, 1])
        axis_vec = jnp.where(
            axis_x[:, None],
            jnp.stack([-jnp.sign(xy_push[:, 0]), jnp.zeros_like(xy_push[:, 0])], axis=1),
            jnp.stack([jnp.zeros_like(xy_push[:, 1]), -jnp.sign(xy_push[:, 1])], axis=1),
        )
        push_mag = trans_scale * push_scale_t
        push_step = jnp.where(push_axis_choice[:, None] < push_square_prob_t, axis_vec, unit_vec) * push_mag[:, None]
        do_push = (push_choice < push_prob_t) & (~do_swap) & (~rot_mask)
        k_move = jnp.where(do_push, k_push, k)
        dxy_move = jnp.where(do_push[:, None], push_step, dxy)

        # "Compact" move: pick a boundary tree and nudge it toward the packing center.
        compact_prob_final = jnp.where(compact_prob_end >= 0.0, compact_prob_end, compact_prob_t)
        current_compact_prob = compact_prob_t + (compact_prob_final - compact_prob_t) * anneal
        current_compact_prob = jnp.clip(current_compact_prob, 0.0, 1.0)
        compact_choice = jax.random.uniform(subkey_compact, (batch_size,))
        do_compact = (compact_choice < current_compact_prob) & (~do_swap) & (~do_push) & (~rot_mask)

        xy_comp = poses[batch_idx, k_boundary, 0:2]
        dir_comp = pack_center - xy_comp
        norm_comp = jnp.linalg.norm(dir_comp, axis=1, keepdims=True)
        unit_comp = dir_comp / (norm_comp + 1e-12)
        axis_comp = jnp.where(
            use_x[:, None],
            jnp.stack([jnp.sign(dir_comp[:, 0]), jnp.zeros_like(dir_comp[:, 0])], axis=1),
            jnp.stack([jnp.zeros_like(dir_comp[:, 1]), jnp.sign(dir_comp[:, 1])], axis=1),
        )
        compact_axis_choice = jax.random.uniform(subkey_compact_axis, (batch_size,))
        compact_dir = jnp.where(compact_axis_choice[:, None] < compact_square_prob_t, axis_comp, unit_comp)
        compact_step = compact_dir * (trans_scale * compact_scale_t)[:, None]

        k_move = jnp.where(do_compact, k_boundary, k_move)
        dxy_move = jnp.where(do_compact[:, None], compact_step, dxy_move)

        delta_xy = dxy_move * (~rot_mask)[:, None]
        delta_theta = dtheta * rot_mask
        delta = jnp.concatenate([delta_xy, delta_theta[:, None]], axis=1)

        candidate_poses_single = poses.at[batch_idx, k_move].add(delta)
        candidate_poses_single = candidate_poses_single.at[:, :, 2].set(jnp.mod(candidate_poses_single[:, :, 2], 360.0))

        # Teleport move: move a boundary tree into a "pocket" near an interior anchor.
        teleport_prob_final = jnp.where(teleport_prob_end >= 0.0, teleport_prob_end, teleport_prob_t)
        current_teleport_prob = teleport_prob_t + (teleport_prob_final - teleport_prob_t) * anneal
        current_teleport_prob = jnp.clip(current_teleport_prob, 0.0, 1.0)
        teleport_choice = jax.random.uniform(subkey_teleport, (batch_size,))
        do_teleport = (teleport_choice < current_teleport_prob) & (~do_swap)

        # Anchor selection: bias toward trees closer to the packing center.
        centers_all = poses[:, :, 0:2]
        d_to_center = centers_all - pack_center[:, None, :]
        dist2 = jnp.sum(d_to_center * d_to_center, axis=2)
        dist = jnp.sqrt(dist2 + 1e-12)

        min_x = jnp.min(bboxes[:, :, 0], axis=1)
        min_y = jnp.min(bboxes[:, :, 1], axis=1)
        max_x = jnp.max(bboxes[:, :, 2], axis=1)
        max_y = jnp.max(bboxes[:, :, 3], axis=1)
        side = jnp.maximum(max_x - min_x, max_y - min_y)
        dist_norm = dist / (side[:, None] + 1e-6)
        logits_anchor = -dist_norm * teleport_anchor_beta_t

        keys_anchor = jax.random.split(subkey_teleport_anchor, batch_size)
        anchor_idx = jax.vmap(lambda kk, logit: jax.random.categorical(kk, logit))(keys_anchor, logits_anchor)
        anchor_xy = poses[batch_idx, anchor_idx, 0:2]

        phi_tries = jax.random.uniform(
            subkey_teleport_phi,
            (teleport_tries, batch_size),
            minval=0.0,
            maxval=2.0 * math.pi,
        )
        noise_tries = jax.random.normal(subkey_teleport_noise, (teleport_tries, batch_size, 2))
        ring_r = (2.0 * radius) * teleport_ring_mult_t
        jitter = noise_tries * (radius * teleport_jitter_t)

        def _attempt_teleport(_: None) -> tuple[jax.Array, jax.Array]:
            found0 = (~do_teleport).astype(bool)
            pose0 = poses[batch_idx, k_boundary]

            def body_fn(t: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
                found, pose_sel = carry
                phi = phi_tries[t]
                unit = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=1)
                cand_xy = anchor_xy + ring_r * unit + jitter[t]
                cand_pose = pose_sel.at[:, 0:2].set(cand_xy)

                pose_k = cand_pose
                new_bbox_k_padded = jax.vmap(lambda p: _aabb_for_pose(p, padded=True))(pose_k)
                cand_bboxes_padded = bboxes_padded.at[batch_idx, k_boundary].set(new_bbox_k_padded)
                new_cell_k = jax.vmap(_cells_for_xy)(pose_k[:, 0:2])
                cand_cells = cells.at[batch_idx, k_boundary].set(new_cell_k)
                cand_poses = poses.at[batch_idx, k_boundary].set(pose_k)

                do_test = do_teleport & (~found)

                def _check_one(p, bbp, cc, idx, test):
                    return jax.lax.cond(
                        test,
                        lambda: _check_collision_for_index_cached(p, bbp, cc, idx),
                        lambda: jnp.array(True),
                    )

                coll = jax.vmap(_check_one)(cand_poses, cand_bboxes_padded, cand_cells, k_boundary, do_test)
                ok = (~coll) & do_test
                pose_sel = jnp.where(ok[:, None], cand_pose, pose_sel)
                found = found | ok
                return found, pose_sel

            found, pose_sel = jax.lax.fori_loop(0, teleport_tries, body_fn, (found0, pose0))
            success = do_teleport & found
            return success, pose_sel

        teleport_success, teleport_pose_k = jax.lax.cond(
            jnp.any(do_teleport),
            _attempt_teleport,
            lambda _: (jnp.zeros((batch_size,), dtype=bool), poses[batch_idx, k_boundary]),
            operand=None,
        )

        candidate_poses_teleport = poses.at[batch_idx, k_boundary].set(teleport_pose_k)
        candidate_poses_single = jnp.where(
            teleport_success[:, None, None], candidate_poses_teleport, candidate_poses_single
        )
        k_move = jnp.where(teleport_success, k_boundary, k_move)
        rot_mask = rot_mask & (~teleport_success)

        # Swap move (k1, k2) with k2 != k1 (when n_trees > 1).
        k1 = k_rand
        if n_trees > 1:
            k2_raw = jax.random.randint(subkey_k2, (batch_size,), 0, n_trees - 1)
            k2 = jnp.where(k2_raw >= k1, k2_raw + 1, k2_raw)
        else:
            k2 = k1
        pose1 = poses[batch_idx, k1]
        pose2 = poses[batch_idx, k2]
        candidate_poses_swap = poses.at[batch_idx, k1].set(pose2)
        candidate_poses_swap = candidate_poses_swap.at[batch_idx, k2].set(pose1)

        candidate_poses = jnp.where(do_swap[:, None, None], candidate_poses_swap, candidate_poses_single)

        # --- Cache updates (AABB + spatial hashing grid) for candidate.
        pose_k = candidate_poses_single[batch_idx, k_move]
        new_bbox_k = jax.vmap(lambda p: _aabb_for_pose(p, padded=False))(pose_k)
        new_bbox_k_padded = jax.vmap(lambda p: _aabb_for_pose(p, padded=True))(pose_k)
        candidate_bboxes_single = bboxes.at[batch_idx, k_move].set(new_bbox_k)
        candidate_bboxes_padded_single = bboxes_padded.at[batch_idx, k_move].set(new_bbox_k_padded)

        new_cell_k = jax.vmap(_cells_for_xy)(pose_k[:, 0:2])
        candidate_cells_single = cells.at[batch_idx, k_move].set(new_cell_k)

        bbox1 = bboxes[batch_idx, k1]
        bbox2 = bboxes[batch_idx, k2]
        candidate_bboxes_swap = bboxes.at[batch_idx, k1].set(bbox2)
        candidate_bboxes_swap = candidate_bboxes_swap.at[batch_idx, k2].set(bbox1)

        bbox1p = bboxes_padded[batch_idx, k1]
        bbox2p = bboxes_padded[batch_idx, k2]
        candidate_bboxes_padded_swap = bboxes_padded.at[batch_idx, k1].set(bbox2p)
        candidate_bboxes_padded_swap = candidate_bboxes_padded_swap.at[batch_idx, k2].set(bbox1p)

        cell1 = cells[batch_idx, k1]
        cell2 = cells[batch_idx, k2]
        candidate_cells_swap = cells.at[batch_idx, k1].set(cell2)
        candidate_cells_swap = candidate_cells_swap.at[batch_idx, k2].set(cell1)

        candidate_bboxes = jnp.where(do_swap[:, None, None], candidate_bboxes_swap, candidate_bboxes_single)
        candidate_bboxes_padded = jnp.where(
            do_swap[:, None, None], candidate_bboxes_padded_swap, candidate_bboxes_padded_single
        )
        candidate_cells = jnp.where(do_swap[:, None, None], candidate_cells_swap, candidate_cells_single)

        # 2. Check Constraints
        # Only the moved tree can introduce a new overlap, so we check one-vs-all.
        is_colliding_single = jax.vmap(_check_collision_for_index_cached)(
            candidate_poses_single,
            candidate_bboxes_padded_single,
            candidate_cells_single,
            k_move,
        )
        is_colliding = jnp.where(do_swap, current_colliding, is_colliding_single)

        # 3. Calculate Score (+ optional overlap penalty)
        candidate_score = jax.vmap(score_from_bboxes_fn)(candidate_bboxes)

        def _update_penalty() -> jax.Array:
            old_k = jax.vmap(lambda p, idx: _pair_penalty_for_index(p[:, :2], idx))(poses, k_move)
            new_k = jax.vmap(lambda p, idx: _pair_penalty_for_index(p[:, :2], idx))(candidate_poses, k_move)
            return current_penalty + (new_k - old_k) * (~do_swap).astype(current_penalty.dtype)

        candidate_penalty = jax.lax.cond(overlap_lambda_t > 0.0, _update_penalty, lambda: current_penalty)
        current_energy = current_score + overlap_lambda_t * current_penalty
        candidate_energy = candidate_score + overlap_lambda_t * candidate_penalty

        # 4. Metropolis Criterion
        delta = candidate_energy - current_energy

        key, subkey_accept = jax.random.split(key)
        r = jax.random.uniform(subkey_accept, (batch_size,))

        # Acceptance condition:
        # If !colliding AND (delta < 0 OR r < exp(-delta/T))
        should_accept = (delta < 0) | (r < jnp.exp(-delta / (temp_eff + 1e-12)))
        if not allow_collisions:
            should_accept = should_accept & (~is_colliding)

        # Update state where accepted
        new_poses = jnp.where(should_accept[:, None, None], candidate_poses, poses)
        new_bboxes = jnp.where(should_accept[:, None, None], candidate_bboxes, bboxes)
        new_bboxes_padded = jnp.where(should_accept[:, None, None], candidate_bboxes_padded, bboxes_padded)
        new_cells = jnp.where(should_accept[:, None, None], candidate_cells, cells)
        new_score = jnp.where(should_accept, candidate_score, current_score)
        new_penalty = jnp.where(should_accept, candidate_penalty, current_penalty)
        new_colliding = jnp.where(should_accept, is_colliding, current_colliding)

        # Update best
        improved = (new_score < best_score) & (~new_colliding)
        new_best_poses = jnp.where(improved[:, None, None], new_poses, best_poses)
        new_best_score = jnp.where(improved, new_score, best_score)

        # --- Reheating update (for next step)
        stall_next = jnp.where(improved, jnp.zeros_like(stall_steps), stall_steps + jnp.int32(1))
        reheat_next = reheat_level * reheat_decay_t
        trigger_reheat = reheat_enabled & (stall_next >= reheat_patience_t)
        stall_next = jnp.where(trigger_reheat, jnp.zeros_like(stall_next), stall_next)
        reheat_next = jnp.where(trigger_reheat, reheat_factor_t, reheat_next)

        # --- Adaptive step-size (per batch element) using acceptance-rate EMA.
        attempt = ~do_swap
        attempt_rot = attempt & rot_mask
        attempt_trans = attempt & (~rot_mask)
        accept_f = should_accept.astype(new_score.dtype)
        metrics_enabled = adapt_sigma_t | adapt_rot_prob_t
        acc_trans_prop = jnp.where(
            attempt_trans,
            acc_trans_ema + adapt_alpha_t * (accept_f - acc_trans_ema),
            acc_trans_ema,
        )
        acc_rot_prop = jnp.where(
            attempt_rot,
            acc_rot_ema + adapt_alpha_t * (accept_f - acc_rot_ema),
            acc_rot_ema,
        )
        acc_trans_next = jnp.where(metrics_enabled, acc_trans_prop, acc_trans_ema)
        acc_rot_next = jnp.where(metrics_enabled, acc_rot_prop, acc_rot_ema)

        trans_sigma_prop = jnp.clip(
            trans_sigma_dyn * jnp.exp(adapt_rate_t * (acc_trans_next - accept_target_t)),
            trans_sigma_min_t,
            trans_sigma_max_t,
        )
        trans_sigma_next = jnp.where(attempt_trans, trans_sigma_prop, trans_sigma_dyn)
        trans_sigma_next = jnp.where(adapt_sigma_t, trans_sigma_next, trans_sigma_dyn)

        rot_sigma_prop = jnp.clip(
            rot_sigma_dyn * jnp.exp(adapt_rate_t * (acc_rot_next - accept_target_t)),
            rot_sigma_min_t,
            rot_sigma_max_t,
        )
        rot_sigma_next = jnp.where(attempt_rot, rot_sigma_prop, rot_sigma_dyn)
        rot_sigma_next = jnp.where(adapt_sigma_t, rot_sigma_next, rot_sigma_dyn)

        rot_prob_prop = (
            rot_prob_dyn
            + rot_prob_pull_t * (rot_prob_base - rot_prob_dyn)
            + rot_prob_adapt_rate_t * (acc_rot_next - acc_trans_next)
        )
        rot_prob_prop = jnp.clip(rot_prob_prop, rot_prob_min_t, rot_prob_max_t)
        rot_prob_next = jnp.where(adapt_rot_prob_t, rot_prob_prop, rot_prob_base)

        return (
            key,
            new_poses,
            new_bboxes,
            new_bboxes_padded,
            new_cells,
            new_score,
            new_penalty,
            new_colliding,
            new_best_poses,
            new_best_score,
            trans_sigma_next,
            rot_sigma_next,
            rot_prob_next,
            acc_trans_next,
            acc_rot_next,
            stall_next,
            reheat_next,
            current_temp,
        ), (new_score, is_colliding)

    # Init State
    batch_size = initial_poses.shape[0]
    initial_bboxes = jax.vmap(lambda p: aabb_for_poses(p, padded=False))(initial_poses)
    initial_bboxes_padded = jax.vmap(lambda p: aabb_for_poses(p, padded=True))(initial_poses)
    initial_cells = jax.vmap(lambda p: _cells_for_xy(p[:, 0:2]))(initial_poses)
    initial_scores = jax.vmap(score_from_bboxes_fn)(initial_bboxes)
    initial_penalty = jax.lax.cond(
        overlap_lambda_t > 0.0,
        lambda: jax.vmap(lambda p: _total_penalty(p[:, :2]))(initial_poses),
        lambda: jnp.zeros((batch_size,), dtype=initial_poses.dtype),
    )
    initial_colliding = jnp.zeros((batch_size,), dtype=bool)
    initial_trans_sigma = jnp.ones((batch_size,), dtype=initial_poses.dtype) * trans_sigma_eff
    initial_rot_sigma = jnp.ones((batch_size,), dtype=initial_poses.dtype) * rot_sigma_eff
    initial_rot_prob = jnp.ones((batch_size,), dtype=initial_poses.dtype) * jnp.asarray(
        rot_prob, dtype=initial_poses.dtype
    )
    initial_acc_trans = jnp.ones((batch_size,), dtype=initial_poses.dtype) * accept_target_t
    initial_acc_rot = jnp.ones((batch_size,), dtype=initial_poses.dtype) * accept_target_t
    initial_stall = jnp.zeros((batch_size,), dtype=jnp.int32)
    initial_reheat = jnp.zeros((batch_size,), dtype=initial_poses.dtype)

    init_state = (
        random_key,
        initial_poses,
        initial_bboxes,
        initial_bboxes_padded,
        initial_cells,
        initial_scores,
        initial_penalty,
        initial_colliding,
        initial_poses,
        initial_scores,
        initial_trans_sigma,
        initial_rot_sigma,
        initial_rot_prob,
        initial_acc_trans,
        initial_acc_rot,
        initial_stall,
        initial_reheat,
        t_start,
    )

    final_state, history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))

    _, _, _, _, _, _, _, _, best_poses, best_score, *_ = final_state

    return best_poses, best_score


@partial(
    jax.jit,
    static_argnames=["n_steps", "n_trees", "objective", "cooling", "allow_collisions"],
)
def run_sa_blocks_batch(
    random_key,
    n_steps,
    n_trees,
    initial_poses,
    blocks,
    blocks_mask,
    t_start=1.0,
    t_end=0.001,
    trans_sigma=0.1,
    rot_sigma=15.0,
    rot_prob=0.3,
    rot_prob_end=-1.0,
    cooling="geom",
    cooling_power=1.0,
    trans_sigma_nexp=0.0,
    rot_sigma_nexp=0.0,
    sigma_nref=50.0,
    overlap_lambda=0.0,
    adapt_sigma=True,
    accept_target=0.35,
    adapt_alpha=0.05,
    adapt_rate=0.1,
    trans_sigma_min=1e-4,
    trans_sigma_max=5.0,
    rot_sigma_min=1e-3,
    rot_sigma_max=90.0,
    adapt_rot_prob=False,
    rot_prob_adapt_rate=0.02,
    rot_prob_pull=0.02,
    rot_prob_min=0.0,
    rot_prob_max=1.0,
    reheat_patience=200,
    reheat_factor=1.0,
    reheat_decay=0.99,
    allow_collisions=False,
    objective="packing",
):
    """Simulated annealing over *rigid blocks* of trees.

    This applies the same translation/rotation delta to every tree index inside
    the selected block. It is intended as a "meta-model" to reduce degrees of
    freedom (e.g., block_size=2..4) before refining individual trees.

    Args:
        blocks: (B, K) int32 array of tree indices.
        blocks_mask: (B, K) bool array; False entries are ignored (padding).
    """

    base_poly = get_tree_polygon()
    score_from_bboxes_fn = prefix_packing_score_from_bboxes if objective == "prefix" else packing_score_from_bboxes

    # Cache AABB tables in the same dtype as poses (avoids repeated recompute/casts).
    table = TREE_AABB_TABLE.astype(initial_poses.dtype)
    table_padded = TREE_AABB_TABLE_PADDED.astype(initial_poses.dtype)

    # Spatial hashing parameters (broad-phase): cell ~ diameter of bounding circle.
    circle_margin = jnp.asarray(1.0 + 1e-4, dtype=initial_poses.dtype)
    thr2 = jnp.asarray(4.0, dtype=initial_poses.dtype) * TREE_RADIUS2.astype(initial_poses.dtype)
    thr2_coll = thr2 * circle_margin
    radius = jnp.sqrt(TREE_RADIUS2).astype(initial_poses.dtype)
    cell_size = (2.0 * radius) * circle_margin
    inv_cell = jnp.asarray(1.0, dtype=initial_poses.dtype) / (cell_size + jnp.asarray(1e-12, dtype=initial_poses.dtype))

    def _cells_for_xy(xy: jax.Array) -> jax.Array:
        return jnp.floor(xy * inv_cell).astype(jnp.int32)

    def _aabb_for_pose(pose: jax.Array, *, padded: bool) -> jax.Array:
        tab = table_padded if padded else table
        idx = theta_to_aabb_bin(pose[2])
        local = tab[idx]
        xy = pose[0:2]
        xyxy = jnp.concatenate([xy, xy], axis=0)
        return local + xyxy

    def _check_collision_for_index_cached(
        poses_one: jax.Array,
        bboxes_padded_one: jax.Array,
        cells_one: jax.Array,
        idx: jax.Array,
    ) -> jax.Array:
        n = poses_one.shape[0]
        not_self = jnp.arange(n) != idx

        cell_k = cells_one[idx]
        dx = jnp.abs(cells_one[:, 0] - cell_k[0]) <= 1
        dy = jnp.abs(cells_one[:, 1] - cell_k[1]) <= 1
        neighbor = dx & dy & not_self

        bbox_k = bboxes_padded_one[idx]
        bbox_overlap = (
            (bbox_k[2] >= bboxes_padded_one[:, 0])
            & (bboxes_padded_one[:, 2] >= bbox_k[0])
            & (bbox_k[3] >= bboxes_padded_one[:, 1])
            & (bboxes_padded_one[:, 3] >= bbox_k[1])
        )

        centers = poses_one[:, :2]
        center_k = centers[idx]
        d = centers - center_k
        dist2 = jnp.sum(d * d, axis=1)
        circle = dist2 <= thr2_coll

        candidate = neighbor & bbox_overlap & circle

        def _check_candidates() -> jax.Array:
            poly_k = transform_polygon(base_poly, poses_one[idx])

            def _check_one(pose_j: jax.Array, do_test: jax.Array) -> jax.Array:
                return jax.lax.cond(
                    do_test,
                    lambda: polygons_intersect(poly_k, transform_polygon(base_poly, pose_j)),
                    lambda: jnp.array(False),
                )

            hits = jax.vmap(_check_one)(poses_one, candidate)
            return jnp.any(hits)

        return jax.lax.cond(jnp.any(candidate), _check_candidates, lambda: jnp.array(False))

    n_ratio = jnp.asarray(float(n_trees), dtype=initial_poses.dtype) / jnp.asarray(
        sigma_nref, dtype=initial_poses.dtype
    )
    trans_sigma_eff = jnp.asarray(trans_sigma, dtype=initial_poses.dtype) * (
        n_ratio ** jnp.asarray(trans_sigma_nexp, dtype=initial_poses.dtype)
    )
    rot_sigma_eff = jnp.asarray(rot_sigma, dtype=initial_poses.dtype) * (
        n_ratio ** jnp.asarray(rot_sigma_nexp, dtype=initial_poses.dtype)
    )
    overlap_lambda_t = jnp.asarray(overlap_lambda, dtype=initial_poses.dtype)

    adapt_sigma_t = jnp.asarray(adapt_sigma, dtype=bool)
    accept_target_t = jnp.asarray(accept_target, dtype=initial_poses.dtype)
    adapt_alpha_t = jnp.asarray(adapt_alpha, dtype=initial_poses.dtype)
    adapt_rate_t = jnp.asarray(adapt_rate, dtype=initial_poses.dtype)
    trans_sigma_min_t = jnp.asarray(trans_sigma_min, dtype=initial_poses.dtype)
    trans_sigma_max_t = jnp.asarray(trans_sigma_max, dtype=initial_poses.dtype)
    rot_sigma_min_t = jnp.asarray(rot_sigma_min, dtype=initial_poses.dtype)
    rot_sigma_max_t = jnp.asarray(rot_sigma_max, dtype=initial_poses.dtype)

    adapt_rot_prob_t = jnp.asarray(adapt_rot_prob, dtype=bool)
    rot_prob_adapt_rate_t = jnp.asarray(rot_prob_adapt_rate, dtype=initial_poses.dtype)
    rot_prob_pull_t = jnp.asarray(rot_prob_pull, dtype=initial_poses.dtype)
    rot_prob_min_t = jnp.asarray(rot_prob_min, dtype=initial_poses.dtype)
    rot_prob_max_t = jnp.asarray(rot_prob_max, dtype=initial_poses.dtype)

    reheat_patience_t = jnp.asarray(reheat_patience, dtype=jnp.int32)
    reheat_factor_t = jnp.asarray(reheat_factor, dtype=initial_poses.dtype)
    reheat_decay_t = jnp.asarray(reheat_decay, dtype=initial_poses.dtype)
    reheat_enabled = reheat_patience_t > 0

    def _pair_penalty_for_index(poses_xy: jax.Array, idx: jax.Array) -> jax.Array:
        center_k = poses_xy[idx]
        d = poses_xy - center_k
        dist2 = jnp.sum(d * d, axis=1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        mask = (jnp.arange(n_trees) != idx).astype(pen.dtype)
        return jnp.sum((pen * pen) * mask)

    def _total_penalty(poses_xy: jax.Array) -> jax.Array:
        d = poses_xy[:, None, :] - poses_xy[None, :, :]
        dist2 = jnp.sum(d * d, axis=-1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        pen2 = pen * pen
        mask = jnp.triu(jnp.ones_like(pen2), k=1)
        return jnp.sum(pen2 * mask)

    def _rotate_about(xy: jax.Array, *, center: jax.Array, delta_deg: jax.Array) -> jax.Array:
        rad = jnp.deg2rad(delta_deg)[:, None, None]
        c = jnp.cos(rad)
        s = jnp.sin(rad)
        v = xy - center[:, None, :]
        x = v[:, :, 0:1]
        y = v[:, :, 1:2]
        xr = x * c - y * s
        yr = x * s + y * c
        return center[:, None, :] + jnp.concatenate([xr, yr], axis=2)

    def step_fn(state, i):
        (
            key,
            poses,
            bboxes,
            bboxes_padded,
            cells,
            current_score,
            current_penalty,
            current_colliding,
            best_poses,
            best_score,
            trans_sigma_dyn,
            rot_sigma_dyn,
            rot_prob_dyn,
            acc_trans_ema,
            acc_rot_ema,
            stall_steps,
            reheat_level,
            temp,
        ) = state

        # Annealing schedule (match run_sa_batch)
        frac = i / n_steps
        anneal = frac ** jnp.asarray(cooling_power, dtype=initial_poses.dtype)
        if cooling in {"geom", "geometric", "exp", "exponential"}:
            current_temp = t_start * (t_end / t_start) ** anneal
        elif cooling == "linear":
            current_temp = t_start + (t_end - t_start) * anneal
        elif cooling == "log":
            denom = jnp.log1p(jnp.asarray(float(max(n_steps - 1, 1)), dtype=initial_poses.dtype))
            alpha = (t_start / t_end - 1.0) / (denom + 1e-12)
            current_temp = t_start / (1.0 + alpha * jnp.log1p(i.astype(initial_poses.dtype)))
        else:
            current_temp = t_start * (t_end / t_start) ** anneal

        temp_eff = current_temp * (jnp.asarray(1.0, dtype=initial_poses.dtype) + reheat_level)

        rot_prob_final = jnp.where(rot_prob_end >= 0.0, rot_prob_end, rot_prob)
        rot_prob_base = rot_prob + (rot_prob_final - rot_prob) * anneal
        rot_prob_base = jnp.clip(rot_prob_base, 0.0, 1.0)
        current_rot_prob = jnp.where(adapt_rot_prob_t, rot_prob_dyn, rot_prob_base)
        current_rot_prob = jnp.clip(current_rot_prob, rot_prob_min_t, rot_prob_max_t)

        key, subkey_block, subkey_move, subkey_trans, subkey_rot = jax.random.split(key, 5)
        batch_size = poses.shape[0]
        batch_idx = jnp.arange(batch_size)

        n_blocks = blocks.shape[0]
        b = jax.random.randint(subkey_block, (batch_size,), 0, n_blocks)
        blk_idx = blocks[b]  # (batch, K)
        blk_mask = blocks_mask[b]  # (batch, K)

        move_choice = jax.random.uniform(subkey_move, (batch_size,))
        rot_mask = move_choice < current_rot_prob

        eps_xy = jax.random.normal(subkey_trans, (batch_size, 2))
        eps_theta = jax.random.normal(subkey_rot, (batch_size,))
        trans_scale = trans_sigma_dyn * temp_eff
        rot_scale = rot_sigma_dyn * temp_eff
        dxy = eps_xy * trans_scale[:, None]
        dtheta = eps_theta * rot_scale

        # Gather current block poses.
        poses_blk = poses[batch_idx[:, None], blk_idx]  # (batch, K, 3)
        xy = poses_blk[:, :, 0:2]
        theta = poses_blk[:, :, 2]
        mask_f = blk_mask.astype(poses.dtype)
        denom = jnp.maximum(jnp.sum(mask_f, axis=1, keepdims=True), 1.0)
        center = jnp.sum(xy * mask_f[:, :, None], axis=1) / denom  # (batch, 2)

        xy_trans = xy + dxy[:, None, :]
        xy_rot = _rotate_about(xy, center=center, delta_deg=dtheta)
        theta_rot = theta + dtheta[:, None]

        cand_xy = jnp.where(rot_mask[:, None, None], xy_rot, xy_trans)
        cand_theta = jnp.where(rot_mask[:, None], theta_rot, theta)
        cand_theta = jnp.mod(cand_theta, 360.0)

        # Scatter updated block poses back.
        candidate_poses = poses
        candidate_bboxes = bboxes
        candidate_bboxes_padded = bboxes_padded
        candidate_cells = cells
        k_size = blk_idx.shape[1]
        for t in range(k_size):
            idx_t = blk_idx[:, t]
            m_t = blk_mask[:, t]
            safe_idx = jnp.where(m_t, idx_t, 0)
            new_pose = jnp.concatenate([cand_xy[:, t, :], cand_theta[:, t : t + 1]], axis=1)
            cur_pose = candidate_poses[batch_idx, safe_idx]
            out_pose = jnp.where(m_t[:, None], new_pose, cur_pose)
            candidate_poses = candidate_poses.at[batch_idx, safe_idx].set(out_pose)

            new_bbox = jax.vmap(lambda p: _aabb_for_pose(p, padded=False))(new_pose)
            cur_bbox = candidate_bboxes[batch_idx, safe_idx]
            out_bbox = jnp.where(m_t[:, None], new_bbox, cur_bbox)
            candidate_bboxes = candidate_bboxes.at[batch_idx, safe_idx].set(out_bbox)

            new_bbox_padded = jax.vmap(lambda p: _aabb_for_pose(p, padded=True))(new_pose)
            cur_bbox_padded = candidate_bboxes_padded[batch_idx, safe_idx]
            out_bbox_padded = jnp.where(m_t[:, None], new_bbox_padded, cur_bbox_padded)
            candidate_bboxes_padded = candidate_bboxes_padded.at[batch_idx, safe_idx].set(out_bbox_padded)

            new_cell = jax.vmap(_cells_for_xy)(new_pose[:, 0:2])
            cur_cell = candidate_cells[batch_idx, safe_idx]
            out_cell = jnp.where(m_t[:, None], new_cell, cur_cell)
            candidate_cells = candidate_cells.at[batch_idx, safe_idx].set(out_cell)

        # 2. Check Constraints: moved indices only (K one-vs-all checks).
        is_colliding = jnp.zeros((batch_size,), dtype=bool)
        for t in range(k_size):
            idx_t = blk_idx[:, t]
            m_t = blk_mask[:, t]
            safe_idx = jnp.where(m_t, idx_t, 0)
            coll_t = jax.vmap(_check_collision_for_index_cached)(
                candidate_poses,
                candidate_bboxes_padded,
                candidate_cells,
                safe_idx,
            )
            is_colliding = is_colliding | (coll_t & m_t)

        # 3. Calculate Score (+ optional overlap penalty)
        candidate_score = jax.vmap(score_from_bboxes_fn)(candidate_bboxes)

        def _update_penalty() -> jax.Array:
            old_xy = poses[:, :, 0:2]
            new_xy = candidate_poses[:, :, 0:2]
            delta_pen = jnp.zeros((batch_size,), dtype=poses.dtype)
            for t in range(k_size):
                idx_t = blk_idx[:, t]
                m_t = blk_mask[:, t]
                safe_idx = jnp.where(m_t, idx_t, 0)
                old_k = jax.vmap(lambda pxy, idx: _pair_penalty_for_index(pxy, idx))(old_xy, safe_idx)
                new_k = jax.vmap(lambda pxy, idx: _pair_penalty_for_index(pxy, idx))(new_xy, safe_idx)
                delta_pen = delta_pen + (new_k - old_k) * m_t.astype(poses.dtype)
            return current_penalty + delta_pen

        candidate_penalty = jax.lax.cond(overlap_lambda_t > 0.0, _update_penalty, lambda: current_penalty)
        current_energy = current_score + overlap_lambda_t * current_penalty
        candidate_energy = candidate_score + overlap_lambda_t * candidate_penalty

        # 4. Metropolis Criterion
        delta = candidate_energy - current_energy
        key, subkey_accept = jax.random.split(key)
        r = jax.random.uniform(subkey_accept, (batch_size,))
        should_accept = (delta < 0) | (r < jnp.exp(-delta / (temp_eff + 1e-12)))
        if not allow_collisions:
            should_accept = should_accept & (~is_colliding)

        new_poses = jnp.where(should_accept[:, None, None], candidate_poses, poses)
        new_bboxes = jnp.where(should_accept[:, None, None], candidate_bboxes, bboxes)
        new_bboxes_padded = jnp.where(should_accept[:, None, None], candidate_bboxes_padded, bboxes_padded)
        new_cells = jnp.where(should_accept[:, None, None], candidate_cells, cells)
        new_score = jnp.where(should_accept, candidate_score, current_score)
        new_penalty = jnp.where(should_accept, candidate_penalty, current_penalty)
        new_colliding = jnp.where(should_accept, is_colliding, current_colliding)

        improved = (new_score < best_score) & (~new_colliding)
        new_best_poses = jnp.where(improved[:, None, None], new_poses, best_poses)
        new_best_score = jnp.where(improved, new_score, best_score)

        stall_next = jnp.where(improved, jnp.zeros_like(stall_steps), stall_steps + jnp.int32(1))
        reheat_next = reheat_level * reheat_decay_t
        trigger_reheat = reheat_enabled & (stall_next >= reheat_patience_t)
        stall_next = jnp.where(trigger_reheat, jnp.zeros_like(stall_next), stall_next)
        reheat_next = jnp.where(trigger_reheat, reheat_factor_t, reheat_next)

        attempt_rot = rot_mask
        attempt_trans = ~rot_mask
        accept_f = should_accept.astype(new_score.dtype)
        metrics_enabled = adapt_sigma_t | adapt_rot_prob_t
        acc_trans_prop = jnp.where(
            attempt_trans,
            acc_trans_ema + adapt_alpha_t * (accept_f - acc_trans_ema),
            acc_trans_ema,
        )
        acc_rot_prop = jnp.where(
            attempt_rot,
            acc_rot_ema + adapt_alpha_t * (accept_f - acc_rot_ema),
            acc_rot_ema,
        )
        acc_trans_next = jnp.where(metrics_enabled, acc_trans_prop, acc_trans_ema)
        acc_rot_next = jnp.where(metrics_enabled, acc_rot_prop, acc_rot_ema)

        trans_sigma_prop = jnp.clip(
            trans_sigma_dyn * jnp.exp(adapt_rate_t * (acc_trans_next - accept_target_t)),
            trans_sigma_min_t,
            trans_sigma_max_t,
        )
        trans_sigma_next = jnp.where(attempt_trans, trans_sigma_prop, trans_sigma_dyn)
        trans_sigma_next = jnp.where(adapt_sigma_t, trans_sigma_next, trans_sigma_dyn)

        rot_sigma_prop = jnp.clip(
            rot_sigma_dyn * jnp.exp(adapt_rate_t * (acc_rot_next - accept_target_t)),
            rot_sigma_min_t,
            rot_sigma_max_t,
        )
        rot_sigma_next = jnp.where(attempt_rot, rot_sigma_prop, rot_sigma_dyn)
        rot_sigma_next = jnp.where(adapt_sigma_t, rot_sigma_next, rot_sigma_dyn)

        rot_prob_prop = (
            rot_prob_dyn
            + rot_prob_pull_t * (rot_prob_base - rot_prob_dyn)
            + rot_prob_adapt_rate_t * (acc_rot_next - acc_trans_next)
        )
        rot_prob_prop = jnp.clip(rot_prob_prop, rot_prob_min_t, rot_prob_max_t)
        rot_prob_next = jnp.where(adapt_rot_prob_t, rot_prob_prop, rot_prob_base)

        return (
            key,
            new_poses,
            new_bboxes,
            new_bboxes_padded,
            new_cells,
            new_score,
            new_penalty,
            new_colliding,
            new_best_poses,
            new_best_score,
            trans_sigma_next,
            rot_sigma_next,
            rot_prob_next,
            acc_trans_next,
            acc_rot_next,
            stall_next,
            reheat_next,
            current_temp,
        ), (new_score, is_colliding)

    batch_size = initial_poses.shape[0]
    initial_bboxes = jax.vmap(lambda p: aabb_for_poses(p, padded=False))(initial_poses)
    initial_bboxes_padded = jax.vmap(lambda p: aabb_for_poses(p, padded=True))(initial_poses)
    initial_cells = jax.vmap(lambda p: _cells_for_xy(p[:, 0:2]))(initial_poses)
    initial_scores = jax.vmap(score_from_bboxes_fn)(initial_bboxes)
    initial_penalty = jax.lax.cond(
        overlap_lambda_t > 0.0,
        lambda: jax.vmap(lambda p: _total_penalty(p[:, :2]))(initial_poses),
        lambda: jnp.zeros((batch_size,), dtype=initial_poses.dtype),
    )
    initial_colliding = jnp.zeros((batch_size,), dtype=bool)
    initial_trans_sigma = jnp.ones((batch_size,), dtype=initial_poses.dtype) * trans_sigma_eff
    initial_rot_sigma = jnp.ones((batch_size,), dtype=initial_poses.dtype) * rot_sigma_eff
    initial_rot_prob = jnp.ones((batch_size,), dtype=initial_poses.dtype) * jnp.asarray(
        rot_prob, dtype=initial_poses.dtype
    )
    initial_acc_trans = jnp.ones((batch_size,), dtype=initial_poses.dtype) * accept_target_t
    initial_acc_rot = jnp.ones((batch_size,), dtype=initial_poses.dtype) * accept_target_t
    initial_stall = jnp.zeros((batch_size,), dtype=jnp.int32)
    initial_reheat = jnp.zeros((batch_size,), dtype=initial_poses.dtype)
    init_state = (
        random_key,
        initial_poses,
        initial_bboxes,
        initial_bboxes_padded,
        initial_cells,
        initial_scores,
        initial_penalty,
        initial_colliding,
        initial_poses,
        initial_scores,
        initial_trans_sigma,
        initial_rot_sigma,
        initial_rot_prob,
        initial_acc_trans,
        initial_acc_rot,
        initial_stall,
        initial_reheat,
        t_start,
    )
    final_state, history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    _, _, _, _, _, _, _, _, best_poses, best_score, *_ = final_state
    return best_poses, best_score


@partial(
    jax.jit,
    static_argnames=[
        "n_steps",
        "n_trees",
        "objective",
        "policy_config",
        "cooling",
        "proposal",
        "allow_collisions",
        "teleport_tries",
    ],
)
def run_sa_batch_guided(
    random_key,
    n_steps,
    n_trees,
    initial_poses,
    policy_params,
    policy_config,
    t_start=1.0,
    t_end=0.001,
    trans_sigma=0.1,
    rot_sigma=15.0,
    rot_prob=0.3,
    rot_prob_end=-1.0,
    swap_prob=0.0,
    swap_prob_end=-1.0,
    cooling="geom",
    cooling_power=1.0,
    trans_sigma_nexp=0.0,
    rot_sigma_nexp=0.0,
    sigma_nref=50.0,
    proposal="random",
    smart_prob=1.0,
    smart_beta=8.0,
    smart_drift=1.0,
    smart_noise=0.25,
    push_prob=0.1,
    push_scale=1.0,
    push_square_prob=0.5,
    compact_prob=0.0,
    compact_prob_end=-1.0,
    compact_scale=1.0,
    compact_square_prob=0.75,
    teleport_prob=0.0,
    teleport_prob_end=-1.0,
    teleport_tries=4,
    teleport_anchor_beta=6.0,
    teleport_ring_mult=1.02,
    teleport_jitter=0.05,
    overlap_lambda=0.0,
    adapt_sigma=True,
    accept_target=0.35,
    adapt_alpha=0.05,
    adapt_rate=0.1,
    trans_sigma_min=1e-4,
    trans_sigma_max=5.0,
    rot_sigma_min=1e-3,
    rot_sigma_max=90.0,
    adapt_rot_prob=False,
    rot_prob_adapt_rate=0.02,
    rot_prob_pull=0.02,
    rot_prob_min=0.0,
    rot_prob_max=1.0,
    reheat_patience=200,
    reheat_factor=1.0,
    reheat_decay=0.99,
    allow_collisions=False,
    objective="packing",
    policy_prob=1.0,
    policy_pmax=0.05,
    policy_prob_end=-1.0,
    policy_pmax_end=-1.0,
):
    """Runs SA where the proposal is a hybrid: learned policy OR heuristic fallback.

    Proposal selection (per batch element):
      - Compute policy logits over trees.
      - If max softmax prob < `policy_pmax`, fallback to heuristic proposal.
      - Else use policy proposal with probability `policy_prob`, otherwise heuristic.

    The heuristic is the same baseline used by `run_sa_batch` (random k + gaussian move).
    """

    base_poly = get_tree_polygon()
    score_from_bboxes_fn = prefix_packing_score_from_bboxes if objective == "prefix" else packing_score_from_bboxes

    # Cache AABB tables in the same dtype as poses (avoids repeated recompute/casts).
    table = TREE_AABB_TABLE.astype(initial_poses.dtype)
    table_padded = TREE_AABB_TABLE_PADDED.astype(initial_poses.dtype)

    # Spatial hashing parameters (broad-phase): cell ~ diameter of bounding circle.
    circle_margin = jnp.asarray(1.0 + 1e-4, dtype=initial_poses.dtype)
    thr2 = jnp.asarray(4.0, dtype=initial_poses.dtype) * TREE_RADIUS2.astype(initial_poses.dtype)
    thr2_coll = thr2 * circle_margin
    radius = jnp.sqrt(TREE_RADIUS2).astype(initial_poses.dtype)
    cell_size = (2.0 * radius) * circle_margin
    inv_cell = jnp.asarray(1.0, dtype=initial_poses.dtype) / (cell_size + jnp.asarray(1e-12, dtype=initial_poses.dtype))

    def _cells_for_xy(xy: jax.Array) -> jax.Array:
        return jnp.floor(xy * inv_cell).astype(jnp.int32)

    def _aabb_for_pose(pose: jax.Array, *, padded: bool) -> jax.Array:
        tab = table_padded if padded else table
        idx = theta_to_aabb_bin(pose[2])
        local = tab[idx]
        xy = pose[0:2]
        xyxy = jnp.concatenate([xy, xy], axis=0)
        return local + xyxy

    def _check_collision_for_index_cached(
        poses_one: jax.Array,
        bboxes_padded_one: jax.Array,
        cells_one: jax.Array,
        idx: jax.Array,
    ) -> jax.Array:
        n = poses_one.shape[0]
        not_self = jnp.arange(n) != idx

        cell_k = cells_one[idx]
        dx = jnp.abs(cells_one[:, 0] - cell_k[0]) <= 1
        dy = jnp.abs(cells_one[:, 1] - cell_k[1]) <= 1
        neighbor = dx & dy & not_self

        bbox_k = bboxes_padded_one[idx]
        bbox_overlap = (
            (bbox_k[2] >= bboxes_padded_one[:, 0])
            & (bboxes_padded_one[:, 2] >= bbox_k[0])
            & (bbox_k[3] >= bboxes_padded_one[:, 1])
            & (bboxes_padded_one[:, 3] >= bbox_k[1])
        )

        centers = poses_one[:, :2]
        center_k = centers[idx]
        d = centers - center_k
        dist2 = jnp.sum(d * d, axis=1)
        circle = dist2 <= thr2_coll

        candidate = neighbor & bbox_overlap & circle

        def _check_candidates() -> jax.Array:
            poly_k = transform_polygon(base_poly, poses_one[idx])

            def _check_one(pose_j: jax.Array, do_test: jax.Array) -> jax.Array:
                return jax.lax.cond(
                    do_test,
                    lambda: polygons_intersect(poly_k, transform_polygon(base_poly, pose_j)),
                    lambda: jnp.array(False),
                )

            hits = jax.vmap(_check_one)(poses_one, candidate)
            return jnp.any(hits)

        return jax.lax.cond(jnp.any(candidate), _check_candidates, lambda: jnp.array(False))

    n_ratio = jnp.asarray(float(n_trees), dtype=initial_poses.dtype) / jnp.asarray(
        sigma_nref, dtype=initial_poses.dtype
    )
    trans_sigma_eff = jnp.asarray(trans_sigma, dtype=initial_poses.dtype) * (
        n_ratio ** jnp.asarray(trans_sigma_nexp, dtype=initial_poses.dtype)
    )
    rot_sigma_eff = jnp.asarray(rot_sigma, dtype=initial_poses.dtype) * (
        n_ratio ** jnp.asarray(rot_sigma_nexp, dtype=initial_poses.dtype)
    )
    overlap_lambda_t = jnp.asarray(overlap_lambda, dtype=initial_poses.dtype)

    push_prob_t = jnp.asarray(push_prob, dtype=initial_poses.dtype)
    push_scale_t = jnp.asarray(push_scale, dtype=initial_poses.dtype)
    push_square_prob_t = jnp.asarray(push_square_prob, dtype=initial_poses.dtype)

    compact_prob_t = jnp.asarray(compact_prob, dtype=initial_poses.dtype)
    compact_scale_t = jnp.asarray(compact_scale, dtype=initial_poses.dtype)
    compact_square_prob_t = jnp.asarray(compact_square_prob, dtype=initial_poses.dtype)

    teleport_prob_t = jnp.asarray(teleport_prob, dtype=initial_poses.dtype)
    teleport_anchor_beta_t = jnp.asarray(teleport_anchor_beta, dtype=initial_poses.dtype)
    teleport_ring_mult_t = jnp.asarray(teleport_ring_mult, dtype=initial_poses.dtype)
    teleport_jitter_t = jnp.asarray(teleport_jitter, dtype=initial_poses.dtype)

    adapt_sigma_t = jnp.asarray(adapt_sigma, dtype=bool)
    accept_target_t = jnp.asarray(accept_target, dtype=initial_poses.dtype)
    adapt_alpha_t = jnp.asarray(adapt_alpha, dtype=initial_poses.dtype)
    adapt_rate_t = jnp.asarray(adapt_rate, dtype=initial_poses.dtype)
    trans_sigma_min_t = jnp.asarray(trans_sigma_min, dtype=initial_poses.dtype)
    trans_sigma_max_t = jnp.asarray(trans_sigma_max, dtype=initial_poses.dtype)
    rot_sigma_min_t = jnp.asarray(rot_sigma_min, dtype=initial_poses.dtype)
    rot_sigma_max_t = jnp.asarray(rot_sigma_max, dtype=initial_poses.dtype)

    adapt_rot_prob_t = jnp.asarray(adapt_rot_prob, dtype=bool)
    rot_prob_adapt_rate_t = jnp.asarray(rot_prob_adapt_rate, dtype=initial_poses.dtype)
    rot_prob_pull_t = jnp.asarray(rot_prob_pull, dtype=initial_poses.dtype)
    rot_prob_min_t = jnp.asarray(rot_prob_min, dtype=initial_poses.dtype)
    rot_prob_max_t = jnp.asarray(rot_prob_max, dtype=initial_poses.dtype)

    reheat_patience_t = jnp.asarray(reheat_patience, dtype=jnp.int32)
    reheat_factor_t = jnp.asarray(reheat_factor, dtype=initial_poses.dtype)
    reheat_decay_t = jnp.asarray(reheat_decay, dtype=initial_poses.dtype)
    reheat_enabled = reheat_patience_t > 0

    def _pair_penalty_for_index(poses_xy: jax.Array, idx: jax.Array) -> jax.Array:
        center_k = poses_xy[idx]
        d = poses_xy - center_k
        dist2 = jnp.sum(d * d, axis=1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        mask = (jnp.arange(n_trees) != idx).astype(pen.dtype)
        return jnp.sum((pen * pen) * mask)

    def _total_penalty(poses_xy: jax.Array) -> jax.Array:
        d = poses_xy[:, None, :] - poses_xy[None, :, :]
        dist2 = jnp.sum(d * d, axis=-1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        pen2 = pen * pen
        mask = jnp.triu(jnp.ones_like(pen2), k=1)
        return jnp.sum(pen2 * mask)

    def _select_bbox_inward(subkey: jax.Array, bboxes: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        min_x = jnp.min(bboxes[:, 0])
        min_y = jnp.min(bboxes[:, 1])
        max_x = jnp.max(bboxes[:, 2])
        max_y = jnp.max(bboxes[:, 3])
        width = max_x - min_x
        height = max_y - min_y
        use_x = width >= height

        slack_x = jnp.minimum(bboxes[:, 0] - min_x, max_x - bboxes[:, 2])
        slack_y = jnp.minimum(bboxes[:, 1] - min_y, max_y - bboxes[:, 3])
        slack = jnp.where(use_x, slack_x, slack_y)
        slack = jnp.maximum(slack, 0.0)
        scale = jnp.maximum(jnp.where(use_x, width, height), 1e-6)
        logits = -(slack / scale) * jnp.asarray(smart_beta, dtype=bboxes.dtype)
        idx = jax.random.categorical(subkey, logits)
        center = jnp.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=bboxes.dtype)
        return idx, center, use_x

    def step_fn(state, i):
        (
            key,
            poses,
            bboxes,
            bboxes_padded,
            cells,
            current_score,
            current_penalty,
            current_colliding,
            best_poses,
            best_score,
            trans_sigma_dyn,
            rot_sigma_dyn,
            rot_prob_dyn,
            acc_trans_ema,
            acc_rot_ema,
            stall_steps,
            reheat_level,
            temp,
        ) = state

        frac = i / n_steps
        anneal = frac ** jnp.asarray(cooling_power, dtype=initial_poses.dtype)
        if cooling in {"geom", "geometric", "exp", "exponential"}:
            current_temp = t_start * (t_end / t_start) ** anneal
        elif cooling == "linear":
            current_temp = t_start + (t_end - t_start) * anneal
        elif cooling == "log":
            denom = jnp.log1p(jnp.asarray(float(max(n_steps - 1, 1)), dtype=initial_poses.dtype))
            alpha = (t_start / t_end - 1.0) / (denom + 1e-12)
            current_temp = t_start / (1.0 + alpha * jnp.log1p(i.astype(initial_poses.dtype)))
        else:
            current_temp = t_start * (t_end / t_start) ** anneal

        temp_eff = current_temp * (jnp.asarray(1.0, dtype=initial_poses.dtype) + reheat_level)

        rot_prob_final = jnp.where(rot_prob_end >= 0.0, rot_prob_end, rot_prob)
        rot_prob_base = rot_prob + (rot_prob_final - rot_prob) * anneal
        rot_prob_base = jnp.clip(rot_prob_base, 0.0, 1.0)
        current_rot_prob = jnp.where(adapt_rot_prob_t, rot_prob_dyn, rot_prob_base)
        current_rot_prob = jnp.clip(current_rot_prob, rot_prob_min_t, rot_prob_max_t)

        swap_prob_final = jnp.where(swap_prob_end >= 0.0, swap_prob_end, swap_prob)
        current_swap_prob = swap_prob + (swap_prob_final - swap_prob) * anneal
        current_swap_prob = jnp.clip(current_swap_prob, 0.0, 1.0)

        policy_prob_final = jnp.where(policy_prob_end >= 0.0, policy_prob_end, policy_prob)
        current_policy_prob = policy_prob + (policy_prob_final - policy_prob) * anneal
        current_policy_prob = jnp.clip(current_policy_prob, 0.0, 1.0)

        pmax_final = jnp.where(policy_pmax_end >= 0.0, policy_pmax_end, policy_pmax)
        current_policy_pmax = policy_pmax + (pmax_final - policy_pmax) * anneal
        current_policy_pmax = jnp.clip(current_policy_pmax, 0.0, 1.0)

        batch_size = poses.shape[0]
        batch_idx = jnp.arange(batch_size)

        # --- Heuristic proposal (baseline random SA move)
        (
            key,
            subkey_select,
            subkey_k,
            subkey_k2,
            subkey_swap,
            subkey_rot_choice,
            subkey_trans,
            subkey_rot,
            subkey_gate,
            subkey_push,
            subkey_push_axis,
            subkey_compact,
            subkey_compact_axis,
            subkey_teleport,
            subkey_teleport_anchor,
            subkey_teleport_phi,
            subkey_teleport_noise,
        ) = jax.random.split(key, 17)
        k_rand = jax.random.randint(subkey_k, (batch_size,), 0, n_trees)
        swap_choice = jax.random.uniform(subkey_swap, (batch_size,))
        if n_trees > 1:
            do_swap = swap_choice < current_swap_prob
        else:
            do_swap = jnp.zeros((batch_size,), dtype=bool)
        rot_choice = jax.random.uniform(subkey_rot_choice, (batch_size,))

        eps_xy = jax.random.normal(subkey_trans, (batch_size, 2))
        eps_theta = jax.random.normal(subkey_rot, (batch_size,))
        trans_scale = trans_sigma_dyn * temp_eff
        rot_scale = rot_sigma_dyn * temp_eff

        keys_pick = jax.random.split(subkey_select, batch_size)
        k_boundary, pack_center, use_x = jax.vmap(_select_bbox_inward)(keys_pick, bboxes)
        xy_k = poses[batch_idx, k_boundary, 0:2]
        direction = pack_center - xy_k
        axis_direction = jnp.where(
            use_x[:, None],
            jnp.stack([direction[:, 0], jnp.zeros_like(direction[:, 0])], axis=1),
            jnp.stack([jnp.zeros_like(direction[:, 1]), direction[:, 1]], axis=1),
        )
        norm = jnp.linalg.norm(axis_direction, axis=1, keepdims=True)
        unit_axis = axis_direction / (norm + 1e-12)
        drift = unit_axis * (trans_scale[:, None] * jnp.asarray(smart_drift, dtype=poses.dtype))
        noise = eps_xy * (trans_scale[:, None] * jnp.asarray(smart_noise, dtype=poses.dtype))
        dxy_smart = drift + noise

        dxy_rand = eps_xy * trans_scale[:, None]

        if proposal in {"bbox", "bbox_inward", "inward", "smart"}:
            k_h = k_boundary
            dxy_h = dxy_smart
        elif proposal == "mixed":
            gate = jax.random.uniform(subkey_gate, (batch_size,)) < jnp.asarray(smart_prob, dtype=poses.dtype)
            k_h = jnp.where(gate, k_boundary, k_rand)
            dxy_h = jnp.where(gate[:, None], dxy_smart, dxy_rand)
        else:
            k_h = k_rand
            dxy_h = dxy_rand

        dtheta_h = eps_theta * rot_scale

        rot_mask = rot_choice < current_rot_prob
        delta_xy = dxy_h * (~rot_mask)[:, None]
        delta_theta = dtheta_h * rot_mask
        delta_h = jnp.concatenate([delta_xy, delta_theta[:, None]], axis=1)

        # Optional deterministic "push" move on heuristic branch (translation-only).
        push_choice = jax.random.uniform(subkey_push, (batch_size,))
        push_axis_choice = jax.random.uniform(subkey_push_axis, (batch_size,))
        abs_xy = jnp.abs(poses[:, :, 0:2])
        push_metric = jnp.maximum(abs_xy[:, :, 0], abs_xy[:, :, 1])
        k_push = jnp.argmax(push_metric, axis=1).astype(jnp.int32)
        xy_push = poses[batch_idx, k_push, 0:2]
        norm = jnp.linalg.norm(xy_push, axis=1, keepdims=True)
        unit_vec = -xy_push / (norm + 1e-12)
        axis_x = jnp.abs(xy_push[:, 0]) >= jnp.abs(xy_push[:, 1])
        axis_vec = jnp.where(
            axis_x[:, None],
            jnp.stack([-jnp.sign(xy_push[:, 0]), jnp.zeros_like(xy_push[:, 0])], axis=1),
            jnp.stack([jnp.zeros_like(xy_push[:, 1]), -jnp.sign(xy_push[:, 1])], axis=1),
        )
        push_mag = trans_scale * push_scale_t
        push_step = jnp.where(push_axis_choice[:, None] < push_square_prob_t, axis_vec, unit_vec) * push_mag[:, None]
        do_push = (push_choice < push_prob_t) & (~do_swap) & (~rot_mask)
        k_h = jnp.where(do_push, k_push, k_h)
        delta_push = jnp.concatenate([push_step, jnp.zeros((batch_size, 1), dtype=poses.dtype)], axis=1)
        delta_h = jnp.where(do_push[:, None], delta_push, delta_h)

        # Compact move (heuristic branch): boundary tree toward center.
        compact_prob_final = jnp.where(compact_prob_end >= 0.0, compact_prob_end, compact_prob_t)
        current_compact_prob = compact_prob_t + (compact_prob_final - compact_prob_t) * anneal
        current_compact_prob = jnp.clip(current_compact_prob, 0.0, 1.0)
        compact_choice = jax.random.uniform(subkey_compact, (batch_size,))
        do_compact = (compact_choice < current_compact_prob) & (~do_swap) & (~do_push) & (~rot_mask)

        xy_comp = poses[batch_idx, k_boundary, 0:2]
        dir_comp = pack_center - xy_comp
        norm_comp = jnp.linalg.norm(dir_comp, axis=1, keepdims=True)
        unit_comp = dir_comp / (norm_comp + 1e-12)
        axis_comp = jnp.where(
            use_x[:, None],
            jnp.stack([jnp.sign(dir_comp[:, 0]), jnp.zeros_like(dir_comp[:, 0])], axis=1),
            jnp.stack([jnp.zeros_like(dir_comp[:, 1]), jnp.sign(dir_comp[:, 1])], axis=1),
        )
        compact_axis_choice = jax.random.uniform(subkey_compact_axis, (batch_size,))
        compact_dir = jnp.where(compact_axis_choice[:, None] < compact_square_prob_t, axis_comp, unit_comp)
        compact_step = compact_dir * (trans_scale * compact_scale_t)[:, None]

        k_h = jnp.where(do_compact, k_boundary, k_h)
        delta_compact = jnp.concatenate([compact_step, jnp.zeros((batch_size, 1), dtype=poses.dtype)], axis=1)
        delta_h = jnp.where(do_compact[:, None], delta_compact, delta_h)

        # --- Policy proposal
        logits_b, mean_b = jax.vmap(lambda p: policy_apply(policy_params, p, policy_config))(poses)
        probs_b = jax.nn.softmax(logits_b, axis=1)
        pmax = jnp.max(probs_b, axis=1)

        key, subkey_gate = jax.random.split(key)
        u = jax.random.uniform(subkey_gate, (batch_size,))
        use_policy = (pmax >= current_policy_pmax) & (u < current_policy_prob)

        key, subkey_pick = jax.random.split(key)
        keys_pick = jax.random.split(subkey_pick, batch_size)
        k_p = jax.vmap(lambda kk, logit: jax.random.categorical(kk, logit))(keys_pick, logits_b)

        key, subkey_eps = jax.random.split(key)
        eps = jax.random.normal(subkey_eps, (batch_size, 3))
        mean_sel = mean_b[batch_idx, k_p]
        scales = jnp.stack([trans_scale, trans_scale, rot_scale], axis=1)
        delta_p = mean_sel * temp_eff[:, None] + eps * scales

        # --- Mix
        k = jnp.where(use_policy, k_p, k_h)
        delta = jnp.where(use_policy[:, None], delta_p, delta_h)

        candidate_poses_single = poses.at[batch_idx, k].add(delta)
        candidate_poses_single = candidate_poses_single.at[:, :, 2].set(jnp.mod(candidate_poses_single[:, :, 2], 360.0))

        # Teleport move (global override): boundary tree into pocket near an interior anchor.
        teleport_prob_final = jnp.where(teleport_prob_end >= 0.0, teleport_prob_end, teleport_prob_t)
        current_teleport_prob = teleport_prob_t + (teleport_prob_final - teleport_prob_t) * anneal
        current_teleport_prob = jnp.clip(current_teleport_prob, 0.0, 1.0)
        teleport_choice = jax.random.uniform(subkey_teleport, (batch_size,))
        do_teleport = (teleport_choice < current_teleport_prob) & (~do_swap)

        centers_all = poses[:, :, 0:2]
        d_to_center = centers_all - pack_center[:, None, :]
        dist2 = jnp.sum(d_to_center * d_to_center, axis=2)
        dist = jnp.sqrt(dist2 + 1e-12)
        min_x = jnp.min(bboxes[:, :, 0], axis=1)
        min_y = jnp.min(bboxes[:, :, 1], axis=1)
        max_x = jnp.max(bboxes[:, :, 2], axis=1)
        max_y = jnp.max(bboxes[:, :, 3], axis=1)
        side = jnp.maximum(max_x - min_x, max_y - min_y)
        dist_norm = dist / (side[:, None] + 1e-6)
        logits_anchor = -dist_norm * teleport_anchor_beta_t
        keys_anchor = jax.random.split(subkey_teleport_anchor, batch_size)
        anchor_idx = jax.vmap(lambda kk, logit: jax.random.categorical(kk, logit))(keys_anchor, logits_anchor)
        anchor_xy = poses[batch_idx, anchor_idx, 0:2]

        phi_tries = jax.random.uniform(
            subkey_teleport_phi,
            (teleport_tries, batch_size),
            minval=0.0,
            maxval=2.0 * math.pi,
        )
        noise_tries = jax.random.normal(subkey_teleport_noise, (teleport_tries, batch_size, 2))
        ring_r = (2.0 * radius) * teleport_ring_mult_t
        jitter = noise_tries * (radius * teleport_jitter_t)

        def _attempt_teleport(_: None) -> tuple[jax.Array, jax.Array]:
            found0 = (~do_teleport).astype(bool)
            pose0 = poses[batch_idx, k_boundary]

            def body_fn(t: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
                found, pose_sel = carry
                phi = phi_tries[t]
                unit = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=1)
                cand_xy = anchor_xy + ring_r * unit + jitter[t]
                cand_pose = pose_sel.at[:, 0:2].set(cand_xy)

                pose_k = cand_pose
                new_bbox_k_padded = jax.vmap(lambda p: _aabb_for_pose(p, padded=True))(pose_k)
                cand_bboxes_padded = bboxes_padded.at[batch_idx, k_boundary].set(new_bbox_k_padded)
                new_cell_k = jax.vmap(_cells_for_xy)(pose_k[:, 0:2])
                cand_cells = cells.at[batch_idx, k_boundary].set(new_cell_k)
                cand_poses = poses.at[batch_idx, k_boundary].set(pose_k)

                do_test = do_teleport & (~found)

                def _check_one(p, bbp, cc, idx, test):
                    return jax.lax.cond(
                        test,
                        lambda: _check_collision_for_index_cached(p, bbp, cc, idx),
                        lambda: jnp.array(True),
                    )

                coll = jax.vmap(_check_one)(cand_poses, cand_bboxes_padded, cand_cells, k_boundary, do_test)
                ok = (~coll) & do_test
                pose_sel = jnp.where(ok[:, None], cand_pose, pose_sel)
                found = found | ok
                return found, pose_sel

            found, pose_sel = jax.lax.fori_loop(0, teleport_tries, body_fn, (found0, pose0))
            success = do_teleport & found
            return success, pose_sel

        teleport_success, teleport_pose_k = jax.lax.cond(
            jnp.any(do_teleport),
            _attempt_teleport,
            lambda _: (jnp.zeros((batch_size,), dtype=bool), poses[batch_idx, k_boundary]),
            operand=None,
        )
        candidate_poses_teleport = poses.at[batch_idx, k_boundary].set(teleport_pose_k)
        candidate_poses_single = jnp.where(
            teleport_success[:, None, None], candidate_poses_teleport, candidate_poses_single
        )
        k = jnp.where(teleport_success, k_boundary, k)
        rot_mask = rot_mask & (~teleport_success)

        # Swap move (useful for objective='prefix'): permutes the order without changing geometry.
        k1 = k_rand
        if n_trees > 1:
            k2_raw = jax.random.randint(subkey_k2, (batch_size,), 0, n_trees - 1)
            k2 = jnp.where(k2_raw >= k1, k2_raw + 1, k2_raw)
        else:
            k2 = k1
        pose1 = poses[batch_idx, k1]
        pose2 = poses[batch_idx, k2]
        candidate_poses_swap = poses.at[batch_idx, k1].set(pose2)
        candidate_poses_swap = candidate_poses_swap.at[batch_idx, k2].set(pose1)

        candidate_poses = jnp.where(do_swap[:, None, None], candidate_poses_swap, candidate_poses_single)

        # --- Cache updates (AABB + spatial hashing grid) for candidate.
        pose_k = candidate_poses_single[batch_idx, k]
        new_bbox_k = jax.vmap(lambda p: _aabb_for_pose(p, padded=False))(pose_k)
        new_bbox_k_padded = jax.vmap(lambda p: _aabb_for_pose(p, padded=True))(pose_k)
        candidate_bboxes_single = bboxes.at[batch_idx, k].set(new_bbox_k)
        candidate_bboxes_padded_single = bboxes_padded.at[batch_idx, k].set(new_bbox_k_padded)

        new_cell_k = jax.vmap(_cells_for_xy)(pose_k[:, 0:2])
        candidate_cells_single = cells.at[batch_idx, k].set(new_cell_k)

        bbox1 = bboxes[batch_idx, k1]
        bbox2 = bboxes[batch_idx, k2]
        candidate_bboxes_swap = bboxes.at[batch_idx, k1].set(bbox2)
        candidate_bboxes_swap = candidate_bboxes_swap.at[batch_idx, k2].set(bbox1)

        bbox1p = bboxes_padded[batch_idx, k1]
        bbox2p = bboxes_padded[batch_idx, k2]
        candidate_bboxes_padded_swap = bboxes_padded.at[batch_idx, k1].set(bbox2p)
        candidate_bboxes_padded_swap = candidate_bboxes_padded_swap.at[batch_idx, k2].set(bbox1p)

        cell1 = cells[batch_idx, k1]
        cell2 = cells[batch_idx, k2]
        candidate_cells_swap = cells.at[batch_idx, k1].set(cell2)
        candidate_cells_swap = candidate_cells_swap.at[batch_idx, k2].set(cell1)

        candidate_bboxes = jnp.where(do_swap[:, None, None], candidate_bboxes_swap, candidate_bboxes_single)
        candidate_bboxes_padded = jnp.where(
            do_swap[:, None, None], candidate_bboxes_padded_swap, candidate_bboxes_padded_single
        )
        candidate_cells = jnp.where(do_swap[:, None, None], candidate_cells_swap, candidate_cells_single)

        # --- Constraints: only moved tree can introduce overlap
        is_colliding_single = jax.vmap(_check_collision_for_index_cached)(
            candidate_poses_single,
            candidate_bboxes_padded_single,
            candidate_cells_single,
            k,
        )
        is_colliding = jnp.where(do_swap, current_colliding, is_colliding_single)

        # --- Score (+ optional overlap penalty) + Metropolis
        candidate_score = jax.vmap(score_from_bboxes_fn)(candidate_bboxes)

        def _update_penalty() -> jax.Array:
            old_k = jax.vmap(lambda p, idx: _pair_penalty_for_index(p[:, :2], idx))(poses, k)
            new_k = jax.vmap(lambda p, idx: _pair_penalty_for_index(p[:, :2], idx))(candidate_poses, k)
            return current_penalty + (new_k - old_k) * (~do_swap).astype(current_penalty.dtype)

        candidate_penalty = jax.lax.cond(overlap_lambda_t > 0.0, _update_penalty, lambda: current_penalty)
        current_energy = current_score + overlap_lambda_t * current_penalty
        candidate_energy = candidate_score + overlap_lambda_t * candidate_penalty
        dscore = candidate_energy - current_energy

        key, subkey_accept = jax.random.split(key)
        r = jax.random.uniform(subkey_accept, (batch_size,))
        should_accept = (dscore < 0) | (r < jnp.exp(-dscore / (temp_eff + 1e-12)))
        if not allow_collisions:
            should_accept = should_accept & (~is_colliding)

        new_poses = jnp.where(should_accept[:, None, None], candidate_poses, poses)
        new_bboxes = jnp.where(should_accept[:, None, None], candidate_bboxes, bboxes)
        new_bboxes_padded = jnp.where(should_accept[:, None, None], candidate_bboxes_padded, bboxes_padded)
        new_cells = jnp.where(should_accept[:, None, None], candidate_cells, cells)
        new_score = jnp.where(should_accept, candidate_score, current_score)
        new_penalty = jnp.where(should_accept, candidate_penalty, current_penalty)
        new_colliding = jnp.where(should_accept, is_colliding, current_colliding)

        improved = (new_score < best_score) & (~new_colliding)
        new_best_poses = jnp.where(improved[:, None, None], new_poses, best_poses)
        new_best_score = jnp.where(improved, new_score, best_score)

        stall_next = jnp.where(improved, jnp.zeros_like(stall_steps), stall_steps + jnp.int32(1))
        reheat_next = reheat_level * reheat_decay_t
        trigger_reheat = reheat_enabled & (stall_next >= reheat_patience_t)
        stall_next = jnp.where(trigger_reheat, jnp.zeros_like(stall_next), stall_next)
        reheat_next = jnp.where(trigger_reheat, reheat_factor_t, reheat_next)

        attempt = ~do_swap
        attempt_trans = attempt & (use_policy | (~use_policy & (~rot_mask)))
        attempt_rot = attempt & (use_policy | (~use_policy & rot_mask))
        accept_f = should_accept.astype(new_score.dtype)
        metrics_enabled = adapt_sigma_t | adapt_rot_prob_t
        acc_trans_prop = jnp.where(
            attempt_trans,
            acc_trans_ema + adapt_alpha_t * (accept_f - acc_trans_ema),
            acc_trans_ema,
        )
        acc_rot_prop = jnp.where(
            attempt_rot,
            acc_rot_ema + adapt_alpha_t * (accept_f - acc_rot_ema),
            acc_rot_ema,
        )
        acc_trans_next = jnp.where(metrics_enabled, acc_trans_prop, acc_trans_ema)
        acc_rot_next = jnp.where(metrics_enabled, acc_rot_prop, acc_rot_ema)

        trans_sigma_prop = jnp.clip(
            trans_sigma_dyn * jnp.exp(adapt_rate_t * (acc_trans_next - accept_target_t)),
            trans_sigma_min_t,
            trans_sigma_max_t,
        )
        trans_sigma_next = jnp.where(attempt_trans, trans_sigma_prop, trans_sigma_dyn)
        trans_sigma_next = jnp.where(adapt_sigma_t, trans_sigma_next, trans_sigma_dyn)

        rot_sigma_prop = jnp.clip(
            rot_sigma_dyn * jnp.exp(adapt_rate_t * (acc_rot_next - accept_target_t)),
            rot_sigma_min_t,
            rot_sigma_max_t,
        )
        rot_sigma_next = jnp.where(attempt_rot, rot_sigma_prop, rot_sigma_dyn)
        rot_sigma_next = jnp.where(adapt_sigma_t, rot_sigma_next, rot_sigma_dyn)

        rot_prob_prop = (
            rot_prob_dyn
            + rot_prob_pull_t * (rot_prob_base - rot_prob_dyn)
            + rot_prob_adapt_rate_t * (acc_rot_next - acc_trans_next)
        )
        rot_prob_prop = jnp.clip(rot_prob_prop, rot_prob_min_t, rot_prob_max_t)
        rot_prob_next = jnp.where(adapt_rot_prob_t, rot_prob_prop, rot_prob_base)

        return (
            key,
            new_poses,
            new_bboxes,
            new_bboxes_padded,
            new_cells,
            new_score,
            new_penalty,
            new_colliding,
            new_best_poses,
            new_best_score,
            trans_sigma_next,
            rot_sigma_next,
            rot_prob_next,
            acc_trans_next,
            acc_rot_next,
            stall_next,
            reheat_next,
            current_temp,
        ), (new_score, is_colliding, use_policy, pmax)

    batch_size = initial_poses.shape[0]
    initial_bboxes = jax.vmap(lambda p: aabb_for_poses(p, padded=False))(initial_poses)
    initial_bboxes_padded = jax.vmap(lambda p: aabb_for_poses(p, padded=True))(initial_poses)
    initial_cells = jax.vmap(lambda p: _cells_for_xy(p[:, 0:2]))(initial_poses)
    initial_scores = jax.vmap(score_from_bboxes_fn)(initial_bboxes)
    initial_penalty = jax.lax.cond(
        overlap_lambda_t > 0.0,
        lambda: jax.vmap(lambda p: _total_penalty(p[:, :2]))(initial_poses),
        lambda: jnp.zeros((batch_size,), dtype=initial_poses.dtype),
    )
    initial_colliding = jnp.zeros((batch_size,), dtype=bool)
    initial_trans_sigma = jnp.ones((batch_size,), dtype=initial_poses.dtype) * trans_sigma_eff
    initial_rot_sigma = jnp.ones((batch_size,), dtype=initial_poses.dtype) * rot_sigma_eff
    initial_rot_prob = jnp.ones((batch_size,), dtype=initial_poses.dtype) * jnp.asarray(
        rot_prob, dtype=initial_poses.dtype
    )
    initial_acc_trans = jnp.ones((batch_size,), dtype=initial_poses.dtype) * accept_target_t
    initial_acc_rot = jnp.ones((batch_size,), dtype=initial_poses.dtype) * accept_target_t
    initial_stall = jnp.zeros((batch_size,), dtype=jnp.int32)
    initial_reheat = jnp.zeros((batch_size,), dtype=initial_poses.dtype)
    init_state = (
        random_key,
        initial_poses,
        initial_bboxes,
        initial_bboxes_padded,
        initial_cells,
        initial_scores,
        initial_penalty,
        initial_colliding,
        initial_poses,
        initial_scores,
        initial_trans_sigma,
        initial_rot_sigma,
        initial_rot_prob,
        initial_acc_trans,
        initial_acc_rot,
        initial_stall,
        initial_reheat,
        t_start,
    )
    final_state, _history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    _, _, _, _, _, _, _, _, best_poses, best_score, *_ = final_state
    return best_poses, best_score
