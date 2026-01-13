#!/usr/bin/env python3

"""CLI to generate a competition-formatted `submission.csv`.

This command builds packings for `n=1..nmax` using a configurable pipeline
(lattice baselines, optional refinement, overlap validation, and formatting).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

import santa_packing.lattice as lattice_mod
from santa_packing.config import config_to_argv, default_config_path
from santa_packing.constants import EPS
from santa_packing.geom_np import (
    packing_bbox,
    packing_score,
    polygon_radius,
    shift_poses_to_origin,
    transform_polygon,
)
from santa_packing.lattice import lattice_poses
from santa_packing.postopt_np import repair_overlaps
from santa_packing.scoring import first_overlap_pair, polygons_intersect
from santa_packing.submission_format import (
    fit_xy_in_bounds,
    format_submission_value,
    quantize_for_submission,
)
from santa_packing.tree_data import TREE_POINTS


def _safe_fallback_layout(points: np.ndarray, n: int) -> np.ndarray:
    radius = float(polygon_radius(points))
    spacing = 2.0 * radius * 1.05
    poses = _grid_initial_poses(n, spacing)
    poses = shift_poses_to_origin(points, poses)
    poses[:, 2] = 0.0
    return poses


def _finalize_puzzle(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    seed: int,
    puzzle_n: int,
    overlap_mode: str,
) -> np.ndarray:
    def _scale_about_center_xy(poses_xydeg: np.ndarray, scale: float) -> np.ndarray:
        scaled = np.array(poses_xydeg, dtype=float, copy=True)
        center = np.mean(scaled[:, 0:2], axis=0)
        scaled[:, 0:2] = center[None, :] + (scaled[:, 0:2] - center[None, :]) * float(scale)
        return scaled

    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] != puzzle_n:
        raise ValueError(f"Puzzle {puzzle_n}: expected {puzzle_n} poses, got {poses.shape[0]}")

    poses[:, 2] = np.mod(poses[:, 2], 360.0)
    if not np.isfinite(poses).all():
        poses = _safe_fallback_layout(points, puzzle_n)

    try:
        poses = fit_xy_in_bounds(poses)
        poses = quantize_for_submission(poses)
    except Exception:
        poses = _safe_fallback_layout(points, puzzle_n)
        poses = fit_xy_in_bounds(poses)
        poses = quantize_for_submission(poses)

    strict_touch_ok = overlap_mode in {"strict", "kaggle"}

    pair = first_overlap_pair(points, poses, eps=EPS, mode=overlap_mode)
    if pair is None:
        return poses

    rng = np.random.default_rng(int(seed) + 991 * int(puzzle_n))

    # --- Fast stochastic "detouch": tiny jitter often breaks exact-touch degeneracies
    # without meaningfully affecting the score.
    # For conservative mode, prefer minimal per-tree nudges (repair) over
    # global scaling to preserve score.
    if not strict_touch_ok:
        for attempt in range(3):
            jitter_xy = 2e-6 * (2.0**attempt)
            jitter_deg = 0.05 * (2.0**attempt)
            candidate = np.array(poses, dtype=float, copy=True)
            candidate[:, 0:2] += rng.normal(0.0, jitter_xy, size=(puzzle_n, 2))
            candidate[:, 2] = np.mod(candidate[:, 2] + rng.normal(0.0, jitter_deg, size=(puzzle_n,)), 360.0)
            try:
                candidate = fit_xy_in_bounds(candidate)
                candidate = quantize_for_submission(candidate)
            except Exception:
                continue
            if first_overlap_pair(points, candidate, eps=EPS, mode=overlap_mode) is None:
                return candidate

        # Monotone "detouch": increasing a uniform expansion about the centroid
        # can only increase inter-tree distances. Search the minimal scale that
        # becomes overlap-free *after quantization* to preserve score.
        def _try_scale(scale: float) -> np.ndarray | None:
            candidate = _scale_about_center_xy(poses, float(scale))
            try:
                candidate = fit_xy_in_bounds(candidate)
                candidate = quantize_for_submission(candidate)
            except Exception:
                return None
            if first_overlap_pair(points, candidate, eps=EPS, mode=overlap_mode) is None:
                return candidate
            return None

        # Find the minimal uniform scale (monotone) that becomes overlap-free
        # *after quantization*. This is typically faster and more reliable than
        # long repair loops for touch-heavy packings.
        lo = 1.0
        hi = 1.0005
        scaled = _try_scale(hi)
        # Allow large scales as last-resort to avoid pathological slow repairs
        # on severely overlapping inputs (autofix use-case).
        while scaled is None and hi < 10.0:
            hi *= 1.05
            scaled = _try_scale(hi)

        if scaled is not None:
            best = hi
            for _ in range(24):
                mid = 0.5 * (lo + best)
                if _try_scale(mid) is not None:
                    best = mid
                else:
                    lo = mid
            out = _try_scale(best)
            if out is None:
                out = scaled
            safer = _try_scale(best * 1.0001)
            return safer if safer is not None else out

    for attempt in range(6):
        jitter_xy = 2e-6 * (2.0**attempt)
        jitter_deg = 0.05 * (2.0**attempt)
        candidate = np.array(poses, dtype=float, copy=True)
        candidate[:, 0:2] += rng.normal(0.0, jitter_xy, size=(puzzle_n, 2))
        candidate[:, 2] = np.mod(candidate[:, 2] + rng.normal(0.0, jitter_deg, size=(puzzle_n,)), 360.0)
        try:
            candidate = fit_xy_in_bounds(candidate)
            candidate = quantize_for_submission(candidate)
        except Exception:
            continue
        if first_overlap_pair(points, candidate, eps=EPS, mode=overlap_mode) is None:
            return candidate

    # --- Best-effort repair (small nudges first; escalate budgets before fallback).
    # When `overlap_mode` counts touching as overlap, a per-tree nudge repair is much
    # safer than global scaling (scaling can create new collisions for concave shapes).
    if strict_touch_ok:
        step0 = max(1e-6, 10.0 * EPS)
    else:
        # Larger default step helps break "sliding touch" chains quickly.
        step0 = max(2e-5, 50.0 * EPS)
    if strict_touch_ok:
        pass1_attempts = 5
        pass2_attempts = 3
    else:
        # Keep conservative repairs bounded; prefer scaling fallback over long repair loops.
        pass1_attempts = 2
        pass2_attempts = 1

    for attempt in range(pass1_attempts):
        repaired = repair_overlaps(
            points,
            poses,
            seed=seed + 97 * attempt,
            max_iters=400 + 400 * attempt,
            step_xy=step0 * (2.0**attempt),
            step_deg=0.0,
            overlap_mode=overlap_mode,
        )
        if repaired is None:
            continue
        try:
            repaired = fit_xy_in_bounds(repaired)
            repaired = quantize_for_submission(repaired)
        except Exception:
            continue
        if first_overlap_pair(points, repaired, eps=EPS, mode=overlap_mode) is None:
            return repaired

    # Second pass: allow tiny angle noise (helps when trees are "interlocked" by touch).
    for attempt in range(pass2_attempts):
        repaired = repair_overlaps(
            points,
            poses,
            seed=seed + 997 * attempt,
            max_iters=2000 + 1000 * attempt,
            step_xy=step0 * (2.0 ** (attempt + 2)),
            step_deg=0.1 * (2.0**attempt),
            overlap_mode=overlap_mode,
        )
        if repaired is None:
            continue
        try:
            repaired = fit_xy_in_bounds(repaired)
            repaired = quantize_for_submission(repaired)
        except Exception:
            continue
        if first_overlap_pair(points, repaired, eps=EPS, mode=overlap_mode) is None:
            return repaired

    # --- Cheap detouch: expand away from centroid (last resort).
    def _try_scale(scale: float) -> np.ndarray | None:
        candidate = _scale_about_center_xy(poses, scale)
        try:
            candidate = fit_xy_in_bounds(candidate)
            candidate = quantize_for_submission(candidate)
        except Exception:
            return None
        if first_overlap_pair(points, candidate, eps=EPS, mode=overlap_mode) is None:
            return candidate
        return None

    # Keep the scale list conservative; large scales are score-destructive and can
    # still fail to resolve some concave "interlock" touch cases.
    if strict_touch_ok:
        scales = (
            1.0005,
            1.001,
            1.002,
            1.003,
            1.005,
            1.007,
            1.01,
            1.015,
            1.02,
            1.025,
            1.03,
            1.035,
            1.04,
            1.045,
            1.05,
            1.06,
            1.07,
            1.08,
            1.09,
            1.10,
        )
    else:
        # Allow a wider range for non-strict modes; this is a last resort and only
        # used when all cheaper detouch/repair attempts failed.
        scales = (
            1.0005,
            1.001,
            1.002,
            1.003,
            1.005,
            1.007,
            1.01,
            1.015,
            1.02,
            1.025,
            1.03,
            1.035,
            1.04,
            1.045,
            1.05,
            1.06,
            1.07,
            1.08,
            1.09,
            1.10,
            1.12,
            1.15,
            1.18,
            1.20,
        )

    for scale in scales:
        candidate = _try_scale(scale)
        if candidate is not None:
            safer = _try_scale(scale * 1.0001)
            return safer if safer is not None else candidate

    # Hard fallback: guaranteed-feasible grid.
    fallback = _safe_fallback_layout(points, puzzle_n)
    fallback = fit_xy_in_bounds(fallback)
    fallback = quantize_for_submission(fallback)
    if first_overlap_pair(points, fallback, eps=EPS, mode=overlap_mode) is not None:
        raise RuntimeError(f"Puzzle {puzzle_n}: failed to produce a feasible (non-overlapping) solution")
    return fallback


def _grid_initial_poses(n: int, spacing: float) -> np.ndarray:
    cols = int(math.ceil(math.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


_JAX_AVAILABLE = None


def _run_sa(
    n: int,
    *,
    seed: int,
    batch_size: int,
    n_steps: int,
    trans_sigma: float,
    rot_sigma: float,
    rot_prob: float,
    rot_prob_end: float,
    swap_prob: float,
    swap_prob_end: float,
    push_prob: float,
    push_scale: float,
    push_square_prob: float,
    compact_prob: float,
    compact_prob_end: float,
    compact_scale: float,
    compact_square_prob: float,
    teleport_prob: float,
    teleport_prob_end: float,
    teleport_tries: int,
    teleport_anchor_beta: float,
    teleport_ring_mult: float,
    teleport_jitter: float,
    cooling: str,
    cooling_power: float,
    trans_sigma_nexp: float,
    rot_sigma_nexp: float,
    sigma_nref: float,
    proposal: str = "random",
    smart_prob: float = 1.0,
    smart_beta: float = 8.0,
    smart_drift: float = 1.0,
    smart_noise: float = 0.25,
    overlap_lambda: float = 0.0,
    allow_collisions: bool = False,
    initial_poses: np.ndarray | None = None,
    objective: str = "packing",
) -> np.ndarray | None:
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is False:
        return None
    try:
        import jax
        import jax.numpy as jnp

        _JAX_AVAILABLE = True
    except Exception:
        _JAX_AVAILABLE = False
        return None

    from santa_packing.optimizer import run_sa_batch  # noqa: E402

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2

    if initial_poses is None:
        initial = _grid_initial_poses(n, spacing)
    else:
        initial = np.array(initial_poses, dtype=float)
    initial_batch = jnp.tile(jnp.array(initial)[None, :, :], (batch_size, 1, 1))

    key = jax.random.PRNGKey(seed)
    best_poses, best_scores = run_sa_batch(
        key,
        n_steps,
        n,
        initial_batch,
        trans_sigma=trans_sigma,
        rot_sigma=rot_sigma,
        rot_prob=rot_prob,
        rot_prob_end=rot_prob_end,
        swap_prob=swap_prob,
        swap_prob_end=swap_prob_end,
        push_prob=push_prob,
        push_scale=push_scale,
        push_square_prob=push_square_prob,
        compact_prob=compact_prob,
        compact_prob_end=compact_prob_end,
        compact_scale=compact_scale,
        compact_square_prob=compact_square_prob,
        teleport_prob=teleport_prob,
        teleport_prob_end=teleport_prob_end,
        teleport_tries=teleport_tries,
        teleport_anchor_beta=teleport_anchor_beta,
        teleport_ring_mult=teleport_ring_mult,
        teleport_jitter=teleport_jitter,
        cooling=cooling,
        cooling_power=cooling_power,
        trans_sigma_nexp=trans_sigma_nexp,
        rot_sigma_nexp=rot_sigma_nexp,
        sigma_nref=sigma_nref,
        proposal=proposal,
        smart_prob=smart_prob,
        smart_beta=smart_beta,
        smart_drift=smart_drift,
        smart_noise=smart_noise,
        overlap_lambda=overlap_lambda,
        allow_collisions=allow_collisions,
        objective=objective,
    )
    best_scores.block_until_ready()
    best_idx = int(jnp.argmin(best_scores))
    poses = np.array(best_poses[best_idx])
    return shift_poses_to_origin(points, poses)


def _build_blocks(n: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    if block_size <= 0:
        raise ValueError("block_size must be >= 1")
    n_blocks = (n + block_size - 1) // block_size
    blocks = np.zeros((n_blocks, block_size), dtype=np.int32)
    mask = np.zeros((n_blocks, block_size), dtype=bool)
    idx = 0
    for b in range(n_blocks):
        for t in range(block_size):
            if idx < n:
                blocks[b, t] = idx
                mask[b, t] = True
                idx += 1
            else:
                blocks[b, t] = 0
                mask[b, t] = False
    return blocks, mask


def _blocks_from_lists(blocks_list: list[list[int]], block_size: int) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.zeros((len(blocks_list), block_size), dtype=np.int32)
    mask = np.zeros((len(blocks_list), block_size), dtype=bool)
    for b, items in enumerate(blocks_list):
        for t, idx in enumerate(items[:block_size]):
            blocks[b, t] = int(idx)
            mask[b, t] = True
    return blocks, mask


def _cluster_blocks(poses: np.ndarray, block_size: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy spatial clustering into blocks of size 2..4 (by XY proximity)."""
    poses = np.array(poses, dtype=float)
    n = int(poses.shape[0])
    if n <= 0:
        return np.zeros((0, block_size), dtype=np.int32), np.zeros((0, block_size), dtype=bool)
    if block_size <= 1:
        blocks = [[i] for i in range(n)]
        return _blocks_from_lists(blocks, block_size=1)

    rng = np.random.default_rng(int(seed))
    remaining = list(range(n))
    rng.shuffle(remaining)
    remaining_set = set(remaining)
    xy = poses[:, 0:2]

    blocks_list: list[list[int]] = []
    while remaining_set:
        # Pick an arbitrary remaining index (shuffled order gives deterministic randomness).
        i = None
        while remaining and i is None:
            cand = remaining.pop()
            if cand in remaining_set:
                i = cand
        if i is None:
            break
        remaining_set.remove(i)
        block = [i]

        for _ in range(block_size - 1):
            if not remaining_set:
                break
            center = np.mean(xy[block], axis=0)
            idxs = np.fromiter(remaining_set, dtype=np.int32)
            d2 = np.sum((xy[idxs] - center[None, :]) ** 2, axis=1)
            j = int(idxs[int(np.argmin(d2))])
            remaining_set.remove(j)
            block.append(j)

        blocks_list.append(block)

    # Stable ordering (helps reproducibility when seed fixed).
    blocks_list.sort(key=lambda b: min(b))
    return _blocks_from_lists(blocks_list, block_size=block_size)


def _rotate_xy(xy: np.ndarray, deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    x = xy[:, 0]
    y = xy[:, 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=1)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return convex hull (CCW) for 2D points as (H,2)."""
    pts = np.array(points, dtype=float)
    if pts.shape[0] <= 1:
        return pts

    # Sort by (x, y).
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if not hull:
        return pts[:1]
    return np.stack(hull, axis=0)


def _block_centers_lattice(
    n_blocks: int,
    *,
    super_poly: np.ndarray,
    pattern: str,
    margin: float,
    rotate_deg: float,
) -> np.ndarray:
    if n_blocks <= 0:
        return np.zeros((0, 3), dtype=float)

    step, row_height = lattice_mod._compute_spacing(super_poly, pattern=pattern, rotate_deg=rotate_deg, margin=margin)
    cols = int(math.ceil(math.sqrt(n_blocks)))
    poses = np.zeros((n_blocks, 3), dtype=float)
    if pattern == "hex":
        for i in range(n_blocks):
            row = i // cols
            col = i % cols
            x = col * step + (step / 2.0 if row % 2 == 1 else 0.0)
            y = row * row_height
            poses[i] = (x, y, rotate_deg)
    else:
        for i in range(n_blocks):
            row = i // cols
            col = i % cols
            x = col * step
            y = row * row_height
            poses[i] = (x, y, rotate_deg)
    # Keep translation-only, the block SA owns rotations.
    poses[:, 2] = 0.0
    return poses


def _block_template_initial_poses(
    points: np.ndarray,
    n: int,
    *,
    seed: int,
    block_size: int,
    template_pattern: str,
    template_margin: float,
    template_rotate_deg: float,
    centers_spacing_scale: float = 1.0,
    random_block_rotations: bool = False,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    if block_size <= 1:
        radius = polygon_radius(points)
        spacing = 2.0 * radius * 1.2
        return shift_poses_to_origin(points, _grid_initial_poses(n, spacing))

    template = lattice_poses(
        block_size,
        pattern=str(template_pattern),
        margin=float(template_margin),
        rotate_deg=float(template_rotate_deg),
    )
    template = np.array(template, dtype=float)

    n_blocks = (n + block_size - 1) // block_size
    super_pts = np.vstack([transform_polygon(points, pose) for pose in template])
    super_poly = _convex_hull(super_pts)
    centers_pose = _block_centers_lattice(
        n_blocks,
        super_poly=super_poly,
        pattern=str(template_pattern),
        margin=0.02,
        rotate_deg=0.0,
    )
    centers = centers_pose[:, 0:2] * float(centers_spacing_scale)

    rng = np.random.default_rng(int(seed))
    block_rots = rng.uniform(0.0, 360.0, size=(n_blocks,)) if random_block_rotations else np.zeros((n_blocks,))

    poses = np.zeros((n, 3), dtype=float)
    idx = 0
    for b in range(n_blocks):
        rot = float(block_rots[b])
        xy_rot = _rotate_xy(template[:, 0:2], rot)
        theta = np.mod(template[:, 2] + rot, 360.0)
        for t in range(block_size):
            if idx >= n:
                break
            poses[idx, 0:2] = centers[b] + xy_rot[t]
            poses[idx, 2] = float(theta[t])
            idx += 1

    return shift_poses_to_origin(points, poses)


def _run_sa_blocks_from_initial(
    n: int,
    *,
    seed: int,
    initial_poses: np.ndarray,
    blocks_np: np.ndarray,
    mask_np: np.ndarray,
    batch_size: int,
    n_steps: int,
    trans_sigma: float,
    rot_sigma: float,
    rot_prob: float,
    rot_prob_end: float,
    cooling: str,
    cooling_power: float,
    trans_sigma_nexp: float,
    rot_sigma_nexp: float,
    sigma_nref: float,
    overlap_lambda: float,
    allow_collisions: bool,
    objective: str,
) -> np.ndarray | None:
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is False:
        return None
    try:
        import jax
        import jax.numpy as jnp

        _JAX_AVAILABLE = True
    except Exception:
        _JAX_AVAILABLE = False
        return None

    from santa_packing.optimizer import run_sa_blocks_batch  # noqa: E402

    points = np.array(TREE_POINTS, dtype=float)
    initial = np.array(initial_poses, dtype=float)

    blocks = jnp.array(blocks_np, dtype=jnp.int32)
    mask = jnp.array(mask_np, dtype=bool)
    initial_batch = jnp.tile(jnp.array(initial)[None, :, :], (int(batch_size), 1, 1))

    key = jax.random.PRNGKey(int(seed))
    best_poses, best_scores = run_sa_blocks_batch(
        key,
        int(n_steps),
        int(n),
        initial_batch,
        blocks,
        mask,
        trans_sigma=float(trans_sigma),
        rot_sigma=float(rot_sigma),
        rot_prob=float(rot_prob),
        rot_prob_end=float(rot_prob_end),
        cooling=str(cooling),
        cooling_power=float(cooling_power),
        trans_sigma_nexp=float(trans_sigma_nexp),
        rot_sigma_nexp=float(rot_sigma_nexp),
        sigma_nref=float(sigma_nref),
        overlap_lambda=float(overlap_lambda),
        allow_collisions=bool(allow_collisions),
        objective=str(objective),
    )
    best_scores.block_until_ready()
    best_idx = int(jnp.argmin(best_scores))
    poses = np.array(best_poses[best_idx])
    return shift_poses_to_origin(points, poses)


def _run_sa_blocks_cluster(
    n: int,
    *,
    seed: int,
    initial_poses: np.ndarray,
    block_size: int,
    batch_size: int,
    n_steps: int,
    trans_sigma: float,
    rot_sigma: float,
    rot_prob: float,
    rot_prob_end: float,
    cooling: str,
    cooling_power: float,
    trans_sigma_nexp: float,
    rot_sigma_nexp: float,
    sigma_nref: float,
    overlap_lambda: float,
    allow_collisions: bool,
    objective: str,
) -> np.ndarray | None:
    blocks_np, mask_np = _cluster_blocks(initial_poses, block_size, seed=seed)
    if blocks_np.shape[0] == 0:
        return np.array(initial_poses, dtype=float)
    return _run_sa_blocks_from_initial(
        n,
        seed=seed,
        initial_poses=initial_poses,
        blocks_np=blocks_np,
        mask_np=mask_np,
        batch_size=batch_size,
        n_steps=n_steps,
        trans_sigma=trans_sigma,
        rot_sigma=rot_sigma,
        rot_prob=rot_prob,
        rot_prob_end=rot_prob_end,
        cooling=cooling,
        cooling_power=cooling_power,
        trans_sigma_nexp=trans_sigma_nexp,
        rot_sigma_nexp=rot_sigma_nexp,
        sigma_nref=sigma_nref,
        overlap_lambda=overlap_lambda,
        allow_collisions=allow_collisions,
        objective=objective,
    )


def _run_sa_blocks_template(
    n: int,
    *,
    seed: int,
    block_size: int,
    batch_size: int,
    n_steps: int,
    trans_sigma: float,
    rot_sigma: float,
    rot_prob: float,
    rot_prob_end: float,
    cooling: str,
    cooling_power: float,
    trans_sigma_nexp: float,
    rot_sigma_nexp: float,
    sigma_nref: float,
    overlap_lambda: float,
    allow_collisions: bool,
    objective: str,
    template_pattern: str,
    template_margin: float,
    template_rotate_deg: float,
) -> np.ndarray | None:
    points = np.array(TREE_POINTS, dtype=float)
    initial = _block_template_initial_poses(
        points,
        n,
        seed=seed,
        block_size=block_size,
        template_pattern=template_pattern,
        template_margin=template_margin,
        template_rotate_deg=template_rotate_deg,
    )

    blocks_np, mask_np = _build_blocks(n, block_size)
    return _run_sa_blocks_from_initial(
        n,
        seed=seed,
        initial_poses=initial,
        blocks_np=blocks_np,
        mask_np=mask_np,
        batch_size=batch_size,
        n_steps=n_steps,
        trans_sigma=trans_sigma,
        rot_sigma=rot_sigma,
        rot_prob=rot_prob,
        rot_prob_end=rot_prob_end,
        cooling=cooling,
        cooling_power=cooling_power,
        trans_sigma_nexp=trans_sigma_nexp,
        rot_sigma_nexp=rot_sigma_nexp,
        sigma_nref=sigma_nref,
        overlap_lambda=overlap_lambda,
        allow_collisions=allow_collisions,
        objective=objective,
    )


def _run_sa_guided(
    n: int,
    *,
    model_path: Path,
    seed: int,
    batch_size: int,
    n_steps: int,
    trans_sigma: float,
    rot_sigma: float,
    rot_prob: float,
    rot_prob_end: float,
    swap_prob: float,
    swap_prob_end: float,
    push_prob: float,
    push_scale: float,
    push_square_prob: float,
    compact_prob: float,
    compact_prob_end: float,
    compact_scale: float,
    compact_square_prob: float,
    teleport_prob: float,
    teleport_prob_end: float,
    teleport_tries: int,
    teleport_anchor_beta: float,
    teleport_ring_mult: float,
    teleport_jitter: float,
    cooling: str,
    cooling_power: float,
    trans_sigma_nexp: float,
    rot_sigma_nexp: float,
    sigma_nref: float,
    proposal: str = "random",
    smart_prob: float = 1.0,
    smart_beta: float = 8.0,
    smart_drift: float = 1.0,
    smart_noise: float = 0.25,
    overlap_lambda: float = 0.0,
    allow_collisions: bool = False,
    objective: str,
    initial_poses: np.ndarray | None = None,
    policy_prob: float = 1.0,
    policy_pmax: float = 0.05,
    policy_prob_end: float = -1.0,
    policy_pmax_end: float = -1.0,
) -> np.ndarray | None:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return None

    from santa_packing.geom_np import polygon_radius  # noqa: E402
    from santa_packing.l2o import L2OConfig, load_params_npz  # noqa: E402
    from santa_packing.optimizer import run_sa_batch_guided  # noqa: E402

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2

    if initial_poses is None:
        initial = _grid_initial_poses(n, spacing)
    else:
        initial = np.array(initial_poses, dtype=float)
    initial_batch = jnp.tile(jnp.array(initial)[None, :, :], (batch_size, 1, 1))

    params, meta = load_params_npz(model_path)

    def _meta_bool(value, default: bool) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, np.ndarray) and value.shape == ():
            return bool(value.item())
        return default

    def _meta_float(value, default: float) -> float:
        if isinstance(value, (float, np.floating)):
            return float(value)
        if isinstance(value, (int, np.integer)):
            return float(value)
        if isinstance(value, np.ndarray) and value.shape == ():
            return float(value.item())
        return default

    hidden = int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32
    policy = str(meta.get("policy", "mlp"))
    knn_k = int(meta.get("knn_k", 4)) if hasattr(meta.get("knn_k", 4), "__int__") else 4
    mlp_depth = int(meta.get("mlp_depth", 1)) if hasattr(meta.get("mlp_depth", 1), "__int__") else 1
    gnn_steps = int(meta.get("gnn_steps", 1)) if hasattr(meta.get("gnn_steps", 1), "__int__") else 1
    gnn_attention = _meta_bool(meta.get("gnn_attention", False), False)
    feature_mode = str(meta.get("feature_mode", "raw"))
    action_scale = _meta_float(meta.get("action_scale", 1.0), 1.0)

    config = L2OConfig(
        hidden_size=hidden,
        policy=policy,
        knn_k=knn_k,
        mlp_depth=mlp_depth,
        gnn_steps=gnn_steps,
        gnn_attention=gnn_attention,
        feature_mode=feature_mode,
        action_scale=action_scale,
        action_noise=False,
    )

    key = jax.random.PRNGKey(seed)
    best_poses, best_scores = run_sa_batch_guided(
        key,
        n_steps,
        n,
        initial_batch,
        params,
        config,
        trans_sigma=trans_sigma,
        rot_sigma=rot_sigma,
        rot_prob=rot_prob,
        rot_prob_end=rot_prob_end,
        swap_prob=swap_prob,
        swap_prob_end=swap_prob_end,
        push_prob=push_prob,
        push_scale=push_scale,
        push_square_prob=push_square_prob,
        compact_prob=compact_prob,
        compact_prob_end=compact_prob_end,
        compact_scale=compact_scale,
        compact_square_prob=compact_square_prob,
        teleport_prob=teleport_prob,
        teleport_prob_end=teleport_prob_end,
        teleport_tries=teleport_tries,
        teleport_anchor_beta=teleport_anchor_beta,
        teleport_ring_mult=teleport_ring_mult,
        teleport_jitter=teleport_jitter,
        cooling=cooling,
        cooling_power=cooling_power,
        trans_sigma_nexp=trans_sigma_nexp,
        rot_sigma_nexp=rot_sigma_nexp,
        sigma_nref=sigma_nref,
        proposal=proposal,
        smart_prob=smart_prob,
        smart_beta=smart_beta,
        smart_drift=smart_drift,
        smart_noise=smart_noise,
        overlap_lambda=overlap_lambda,
        allow_collisions=allow_collisions,
        objective=objective,
        policy_prob=policy_prob,
        policy_pmax=policy_pmax,
        policy_prob_end=policy_prob_end,
        policy_pmax_end=policy_pmax_end,
    )
    best_scores.block_until_ready()
    best_idx = int(jnp.argmin(best_scores))
    poses = np.array(best_poses[best_idx])
    return shift_poses_to_origin(points, poses)


def _run_l2o(
    n: int,
    *,
    model_path: Path,
    seed: int,
    steps: int,
    trans_sigma: float,
    rot_sigma: float,
    deterministic: bool,
    initial_poses: np.ndarray | None = None,
) -> np.ndarray | None:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return None

    from santa_packing.geom_np import polygon_radius  # noqa: E402
    from santa_packing.l2o import L2OConfig, load_params_npz, optimize_with_l2o  # noqa: E402

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2
    if initial_poses is None:
        initial = _grid_initial_poses(n, spacing)
    else:
        initial = np.array(initial_poses, dtype=float)

    params, meta = load_params_npz(model_path)
    policy = meta.get("policy", "mlp")
    knn_k = int(meta.get("knn_k", 4)) if hasattr(meta.get("knn_k", 4), "__int__") else 4
    mlp_depth = int(meta.get("mlp_depth", 1)) if hasattr(meta.get("mlp_depth", 1), "__int__") else 1
    gnn_steps = int(meta.get("gnn_steps", 1)) if hasattr(meta.get("gnn_steps", 1), "__int__") else 1
    feature_mode = str(meta.get("feature_mode", "raw"))

    def _meta_bool(value, default: bool) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, np.ndarray) and value.shape == ():
            return bool(value.item())
        return default

    gnn_attention = _meta_bool(meta.get("gnn_attention", False), False)
    hidden = int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32
    config = L2OConfig(
        hidden_size=hidden,
        policy=str(policy),
        knn_k=knn_k,
        mlp_depth=mlp_depth,
        gnn_steps=gnn_steps,
        gnn_attention=gnn_attention,
        feature_mode=feature_mode,
        trans_sigma=trans_sigma,
        rot_sigma=rot_sigma,
        action_noise=not deterministic,
    )
    key = jax.random.PRNGKey(seed)
    poses = optimize_with_l2o(
        key,
        params,
        jnp.array(initial),
        steps,
        config,
    )
    poses = np.array(poses)
    poses = shift_poses_to_origin(points, poses)
    if _has_overlaps(points, poses):
        return None
    return poses


def _has_overlaps(points: np.ndarray, poses: np.ndarray) -> bool:
    poses = np.array(poses, dtype=float)
    if poses.shape[0] <= 1:
        return False
    centers = poses[:, :2]
    rad = float(polygon_radius(points))
    thr2 = (2.0 * rad) ** 2
    polys = [None] * poses.shape[0]
    for i in range(poses.shape[0]):
        polys[i] = transform_polygon(points, poses[i])
    for i in range(poses.shape[0]):
        for j in range(i + 1, poses.shape[0]):
            dx = float(centers[i, 0] - centers[j, 0])
            dy = float(centers[i, 1] - centers[j, 1])
            if dx * dx + dy * dy > thr2:
                continue
            if polygons_intersect(polys[i], polys[j]):
                return True
    return False


def _parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    raw = text.strip()
    if not raw:
        return []
    if raw.lower() in {"none", "off", "false"}:
        return []
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _best_lattice_poses(
    n: int,
    *,
    pattern: str,
    margin: float,
    rotate_deg: float,
    rotate_mode: str,
    rotate_degs: list[float] | None,
) -> np.ndarray:
    if rotate_mode != "constant":
        seq = rotate_degs if rotate_degs else [rotate_deg]
        return lattice_poses(
            n,
            pattern=pattern,
            margin=margin,
            rotate_deg=rotate_deg,
            rotate_mode=str(rotate_mode),
            rotate_degs=seq,
        )

    if not rotate_degs:
        return lattice_poses(n, pattern=pattern, margin=margin, rotate_deg=rotate_deg, rotate_mode="constant")

    points = np.array(TREE_POINTS, dtype=float)
    best_score = float("inf")
    best_poses: np.ndarray | None = None
    for deg in rotate_degs:
        poses = lattice_poses(n, pattern=pattern, margin=margin, rotate_deg=deg, rotate_mode="constant")
        s = packing_score(points, poses)
        if s < best_score:
            best_score = s
            best_poses = poses
    assert best_poses is not None
    return best_poses


def _radial_reorder(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    poses = np.array(poses, dtype=float)
    if poses.shape[0] <= 1:
        return poses

    bbox = packing_bbox(points, poses)
    center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=float)
    d = poses[:, :2] - center[None, :]
    dist2 = np.sum(d * d, axis=1)

    order = np.argsort(dist2, kind="mergesort")
    return poses[order]


def solve_n(
    n: int,
    *,
    seed: int,
    overlap_mode: str = "strict",
    lattice_pattern: str,
    lattice_margin: float,
    lattice_rotate_deg: float,
    lattice_rotate_mode: str,
    lattice_rotate_degs: list[float] | None,
    lattice_post_nmax: int,
    lattice_post_steps: int,
    lattice_post_step_xy: float,
    lattice_post_step_deg: float,
    sa_nmax: int,
    sa_batch_size: int,
    sa_steps: int,
    sa_trans_sigma: float,
    sa_rot_sigma: float,
    sa_rot_prob: float,
    sa_rot_prob_end: float,
    sa_swap_prob: float,
    sa_swap_prob_end: float,
    sa_push_prob: float,
    sa_push_scale: float,
    sa_push_square_prob: float,
    sa_compact_prob: float,
    sa_compact_prob_end: float,
    sa_compact_scale: float,
    sa_compact_square_prob: float,
    sa_teleport_prob: float,
    sa_teleport_prob_end: float,
    sa_teleport_tries: int,
    sa_teleport_anchor_beta: float,
    sa_teleport_ring_mult: float,
    sa_teleport_jitter: float,
    sa_cooling: str,
    sa_cooling_power: float,
    sa_trans_sigma_nexp: float,
    sa_rot_sigma_nexp: float,
    sa_sigma_nref: float,
    sa_proposal: str,
    sa_smart_prob: float,
    sa_smart_beta: float,
    sa_smart_drift: float,
    sa_smart_noise: float,
    sa_overlap_lambda: float,
    sa_allow_collisions: bool,
    sa_objective: str,
    meta_init_model: Path | None,
    heatmap_model: Path | None,
    heatmap_nmax: int,
    heatmap_steps: int,
    l2o_model: Path | None,
    l2o_init: str,
    l2o_nmax: int,
    l2o_steps: int,
    l2o_trans_sigma: float,
    l2o_rot_sigma: float,
    l2o_deterministic: bool,
    refine_nmin: int,
    refine_batch_size: int,
    refine_steps: int,
    refine_trans_sigma: float,
    refine_rot_sigma: float,
    refine_rot_prob: float,
    refine_rot_prob_end: float,
    refine_swap_prob: float,
    refine_swap_prob_end: float,
    refine_push_prob: float,
    refine_push_scale: float,
    refine_push_square_prob: float,
    refine_compact_prob: float,
    refine_compact_prob_end: float,
    refine_compact_scale: float,
    refine_compact_square_prob: float,
    refine_teleport_prob: float,
    refine_teleport_prob_end: float,
    refine_teleport_tries: int,
    refine_teleport_anchor_beta: float,
    refine_teleport_ring_mult: float,
    refine_teleport_jitter: float,
    refine_cooling: str,
    refine_cooling_power: float,
    refine_trans_sigma_nexp: float,
    refine_rot_sigma_nexp: float,
    refine_sigma_nref: float,
    refine_proposal: str,
    refine_smart_prob: float,
    refine_smart_beta: float,
    refine_smart_drift: float,
    refine_smart_noise: float,
    refine_overlap_lambda: float,
    refine_allow_collisions: bool,
    refine_objective: str,
    lns_nmax: int,
    lns_passes: int,
    lns_destroy_k: int,
    lns_destroy_mode: str,
    lns_tabu_tenure: int,
    lns_candidates: int,
    lns_angle_samples: int,
    lns_pad_scale: float,
    lns_group_moves: int,
    lns_group_size: int,
    lns_group_trans_sigma: float,
    lns_group_rot_sigma: float,
    lns_t_start: float,
    lns_t_end: float,
    hc_nmax: int,
    hc_passes: int,
    hc_step_xy: float,
    hc_step_deg: float,
    ga_nmax: int,
    ga_pop: int,
    ga_gens: int,
    ga_elite_frac: float,
    ga_crossover_prob: float,
    ga_mut_sigma_xy: float,
    ga_mut_sigma_deg: float,
    ga_directed_prob: float,
    ga_directed_step_xy: float,
    ga_directed_k: int,
    ga_repair_iters: int,
    ga_hc_passes: int,
    ga_hc_step_xy: float,
    ga_hc_step_deg: float,
    guided_model: Path | None,
    guided_prob: float,
    guided_pmax: float,
    guided_prob_end: float,
    guided_pmax_end: float,
    block_nmax: int,
    block_size: int,
    block_batch_size: int,
    block_steps: int,
    block_trans_sigma: float,
    block_rot_sigma: float,
    block_rot_prob: float,
    block_rot_prob_end: float,
    block_cooling: str,
    block_cooling_power: float,
    block_trans_sigma_nexp: float,
    block_rot_sigma_nexp: float,
    block_sigma_nref: float,
    block_overlap_lambda: float,
    block_allow_collisions: bool,
    block_objective: str,
    block_init: str,
    block_template_pattern: str,
    block_template_margin: float,
    block_template_rotate: float,
) -> np.ndarray:
    """Solve a single puzzle size `n` and return poses as an `(n, 3)` array.

    This is the core pipeline behind `generate_submission`: start from a baseline
    initialization (typically `lattice_*`), optionally apply learned components
    (`meta_init_model`, `heatmap_model`, `l2o_model`, `guided_model`), refine with
    SA (`sa_*` / `refine_*`) and neighborhood search (`lns_*`), and finally return
    poses ready for validation/formatting.

    Parameters are grouped by prefix and match the CLI flags (see `--help`).

    Returns:
        Numpy array shaped `(n, 3)` with columns `(x, y, deg)`.
    """
    base: np.ndarray | None = None

    if heatmap_model is not None and n <= heatmap_nmax:
        try:
            from santa_packing.heatmap_meta import HeatmapConfig, heatmap_search, load_params  # noqa: E402
        except Exception:
            heatmap_model = None
        if heatmap_model is not None:
            params, meta = load_params(heatmap_model)
            config = HeatmapConfig(
                hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
                policy=str(meta.get("policy", "gnn")),
                knn_k=int(meta.get("knn_k", 4)) if hasattr(meta.get("knn_k", 4), "__int__") else 4,
                heatmap_lr=float(meta.get("heatmap_lr", 0.1)),
                trans_sigma=float(meta.get("trans_sigma", 0.2)),
                rot_sigma=float(meta.get("rot_sigma", 10.0)),
                t_start=float(meta.get("t_start", 1.0)),
                t_end=float(meta.get("t_end", 0.001)),
            )
            points = np.array(TREE_POINTS, dtype=float)
            radius = polygon_radius(points)
            spacing = 2.0 * radius * 1.2
            base = _grid_initial_poses(n, spacing)
            rng = np.random.default_rng(seed)
            poses, _ = heatmap_search(params, base, config, heatmap_steps, rng)
            base = poses

    if base is None and l2o_model is not None and n <= l2o_nmax:
        l2o_initial = None
        if l2o_init == "lattice":
            l2o_initial = _best_lattice_poses(
                n,
                pattern=lattice_pattern,
                margin=lattice_margin,
                rotate_deg=lattice_rotate_deg,
                rotate_mode=lattice_rotate_mode,
                rotate_degs=lattice_rotate_degs,
            )
        poses = _run_l2o(
            n,
            model_path=l2o_model,
            seed=seed,
            steps=l2o_steps,
            trans_sigma=l2o_trans_sigma,
            rot_sigma=l2o_rot_sigma,
            deterministic=l2o_deterministic,
            initial_poses=l2o_initial,
        )
        if poses is not None:
            base = poses

    init_override = None
    if base is None and n <= sa_nmax and meta_init_model is not None:
        try:
            import jax.numpy as jnp
        except Exception:
            meta_init_model = None
        if meta_init_model is not None:
            from santa_packing.meta_init import MetaInitConfig, apply_meta_init, load_meta_params  # noqa: E402

            points = np.array(TREE_POINTS, dtype=float)
            radius = polygon_radius(points)
            spacing = 2.0 * radius * 1.2
            grid_base = _grid_initial_poses(n, spacing)
            params, meta = load_meta_params(meta_init_model)
            config = MetaInitConfig(
                hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
                delta_xy=float(meta.get("delta_xy", 0.2)),
                delta_theta=float(meta.get("delta_theta", 10.0)),
            )
            init_override = np.array(apply_meta_init(params, jnp.array(grid_base), config))
            if _has_overlaps(points, init_override):
                init_override = grid_base

    # --- Meta-model blocks: optimize rigid groups before refining individual trees.
    if base is None and block_steps > 0 and block_nmax > 0 and n <= block_nmax and 2 <= block_size <= 4:
        block_initial = init_override
        if block_initial is None:
            block_initial = _best_lattice_poses(
                n,
                pattern=lattice_pattern,
                margin=lattice_margin,
                rotate_deg=lattice_rotate_deg,
                rotate_mode=lattice_rotate_mode,
                rotate_degs=lattice_rotate_degs,
            )

        if block_init == "template":
            block_poses = _run_sa_blocks_template(
                n,
                seed=seed,
                block_size=block_size,
                batch_size=block_batch_size,
                n_steps=block_steps,
                trans_sigma=block_trans_sigma,
                rot_sigma=block_rot_sigma,
                rot_prob=block_rot_prob,
                rot_prob_end=block_rot_prob_end,
                cooling=block_cooling,
                cooling_power=block_cooling_power,
                trans_sigma_nexp=block_trans_sigma_nexp,
                rot_sigma_nexp=block_rot_sigma_nexp,
                sigma_nref=block_sigma_nref,
                overlap_lambda=block_overlap_lambda,
                allow_collisions=block_allow_collisions,
                objective=block_objective,
                template_pattern=block_template_pattern,
                template_margin=block_template_margin,
                template_rotate_deg=block_template_rotate,
            )
        else:
            block_poses = _run_sa_blocks_cluster(
                n,
                seed=seed,
                initial_poses=block_initial,
                block_size=block_size,
                batch_size=block_batch_size,
                n_steps=block_steps,
                trans_sigma=block_trans_sigma,
                rot_sigma=block_rot_sigma,
                rot_prob=block_rot_prob,
                rot_prob_end=block_rot_prob_end,
                cooling=block_cooling,
                cooling_power=block_cooling_power,
                trans_sigma_nexp=block_trans_sigma_nexp,
                rot_sigma_nexp=block_rot_sigma_nexp,
                sigma_nref=block_sigma_nref,
                overlap_lambda=block_overlap_lambda,
                allow_collisions=block_allow_collisions,
                objective=block_objective,
            )
        if block_poses is not None:
            if n <= sa_nmax:
                init_override = block_poses
            else:
                base = block_poses

    if base is None and n <= sa_nmax:
        if guided_model is not None:
            poses = _run_sa_guided(
                n,
                model_path=guided_model,
                seed=seed,
                batch_size=sa_batch_size,
                n_steps=sa_steps,
                trans_sigma=sa_trans_sigma,
                rot_sigma=sa_rot_sigma,
                rot_prob=sa_rot_prob,
                rot_prob_end=sa_rot_prob_end,
                swap_prob=sa_swap_prob,
                swap_prob_end=sa_swap_prob_end,
                push_prob=sa_push_prob,
                push_scale=sa_push_scale,
                push_square_prob=sa_push_square_prob,
                compact_prob=sa_compact_prob,
                compact_prob_end=sa_compact_prob_end,
                compact_scale=sa_compact_scale,
                compact_square_prob=sa_compact_square_prob,
                teleport_prob=sa_teleport_prob,
                teleport_prob_end=sa_teleport_prob_end,
                teleport_tries=sa_teleport_tries,
                teleport_anchor_beta=sa_teleport_anchor_beta,
                teleport_ring_mult=sa_teleport_ring_mult,
                teleport_jitter=sa_teleport_jitter,
                cooling=sa_cooling,
                cooling_power=sa_cooling_power,
                trans_sigma_nexp=sa_trans_sigma_nexp,
                rot_sigma_nexp=sa_rot_sigma_nexp,
                sigma_nref=sa_sigma_nref,
                proposal=sa_proposal,
                smart_prob=sa_smart_prob,
                smart_beta=sa_smart_beta,
                smart_drift=sa_smart_drift,
                smart_noise=sa_smart_noise,
                overlap_lambda=sa_overlap_lambda,
                allow_collisions=sa_allow_collisions,
                initial_poses=init_override,
                objective=sa_objective,
                policy_prob=guided_prob,
                policy_pmax=guided_pmax,
                policy_prob_end=guided_prob_end,
                policy_pmax_end=guided_pmax_end,
            )
        else:
            poses = _run_sa(
                n,
                seed=seed,
                batch_size=sa_batch_size,
                n_steps=sa_steps,
                trans_sigma=sa_trans_sigma,
                rot_sigma=sa_rot_sigma,
                rot_prob=sa_rot_prob,
                rot_prob_end=sa_rot_prob_end,
                swap_prob=sa_swap_prob,
                swap_prob_end=sa_swap_prob_end,
                push_prob=sa_push_prob,
                push_scale=sa_push_scale,
                push_square_prob=sa_push_square_prob,
                compact_prob=sa_compact_prob,
                compact_prob_end=sa_compact_prob_end,
                compact_scale=sa_compact_scale,
                compact_square_prob=sa_compact_square_prob,
                teleport_prob=sa_teleport_prob,
                teleport_prob_end=sa_teleport_prob_end,
                teleport_tries=sa_teleport_tries,
                teleport_anchor_beta=sa_teleport_anchor_beta,
                teleport_ring_mult=sa_teleport_ring_mult,
                teleport_jitter=sa_teleport_jitter,
                cooling=sa_cooling,
                cooling_power=sa_cooling_power,
                trans_sigma_nexp=sa_trans_sigma_nexp,
                rot_sigma_nexp=sa_rot_sigma_nexp,
                sigma_nref=sa_sigma_nref,
                proposal=sa_proposal,
                smart_prob=sa_smart_prob,
                smart_beta=sa_smart_beta,
                smart_drift=sa_smart_drift,
                smart_noise=sa_smart_noise,
                overlap_lambda=sa_overlap_lambda,
                allow_collisions=sa_allow_collisions,
                initial_poses=init_override,
                objective=sa_objective,
            )
        if poses is not None:
            base = poses

    if base is None:
        base = _best_lattice_poses(
            n,
            pattern=lattice_pattern,
            margin=lattice_margin,
            rotate_deg=lattice_rotate_deg,
            rotate_mode=lattice_rotate_mode,
            rotate_degs=lattice_rotate_degs,
        )
        if lattice_post_nmax > 0 and lattice_post_steps > 0 and n <= lattice_post_nmax:
            from santa_packing.postopt_np import hill_climb_boundary  # noqa: E402

            points = np.array(TREE_POINTS, dtype=float)
            base = hill_climb_boundary(
                points,
                base,
                steps=lattice_post_steps,
                step_xy=lattice_post_step_xy,
                step_deg=lattice_post_step_deg,
                seed=seed,
                overlap_mode=str(overlap_mode),
            )

    points = np.array(TREE_POINTS, dtype=float)
    base = _post_optimize(
        points,
        base,
        n=n,
        seed=seed,
        overlap_mode=overlap_mode,
        refine_nmin=refine_nmin,
        refine_batch_size=refine_batch_size,
        refine_steps=refine_steps,
        refine_trans_sigma=refine_trans_sigma,
        refine_rot_sigma=refine_rot_sigma,
        refine_rot_prob=refine_rot_prob,
        refine_rot_prob_end=refine_rot_prob_end,
        refine_swap_prob=refine_swap_prob,
        refine_swap_prob_end=refine_swap_prob_end,
        refine_push_prob=refine_push_prob,
        refine_push_scale=refine_push_scale,
        refine_push_square_prob=refine_push_square_prob,
        refine_compact_prob=refine_compact_prob,
        refine_compact_prob_end=refine_compact_prob_end,
        refine_compact_scale=refine_compact_scale,
        refine_compact_square_prob=refine_compact_square_prob,
        refine_teleport_prob=refine_teleport_prob,
        refine_teleport_prob_end=refine_teleport_prob_end,
        refine_teleport_tries=refine_teleport_tries,
        refine_teleport_anchor_beta=refine_teleport_anchor_beta,
        refine_teleport_ring_mult=refine_teleport_ring_mult,
        refine_teleport_jitter=refine_teleport_jitter,
        refine_cooling=refine_cooling,
        refine_cooling_power=refine_cooling_power,
        refine_trans_sigma_nexp=refine_trans_sigma_nexp,
        refine_rot_sigma_nexp=refine_rot_sigma_nexp,
        refine_sigma_nref=refine_sigma_nref,
        refine_proposal=refine_proposal,
        refine_smart_prob=refine_smart_prob,
        refine_smart_beta=refine_smart_beta,
        refine_smart_drift=refine_smart_drift,
        refine_smart_noise=refine_smart_noise,
        refine_overlap_lambda=refine_overlap_lambda,
        refine_allow_collisions=refine_allow_collisions,
        refine_objective=refine_objective,
        guided_model=guided_model,
        guided_prob=guided_prob,
        guided_pmax=guided_pmax,
        guided_prob_end=guided_prob_end,
        guided_pmax_end=guided_pmax_end,
        lns_nmax=lns_nmax,
        lns_passes=lns_passes,
        lns_destroy_k=lns_destroy_k,
        lns_destroy_mode=lns_destroy_mode,
        lns_tabu_tenure=lns_tabu_tenure,
        lns_candidates=lns_candidates,
        lns_angle_samples=lns_angle_samples,
        lns_pad_scale=lns_pad_scale,
        lns_group_moves=lns_group_moves,
        lns_group_size=lns_group_size,
        lns_group_trans_sigma=lns_group_trans_sigma,
        lns_group_rot_sigma=lns_group_rot_sigma,
        lns_t_start=lns_t_start,
        lns_t_end=lns_t_end,
        hc_nmax=hc_nmax,
        hc_passes=hc_passes,
        hc_step_xy=hc_step_xy,
        hc_step_deg=hc_step_deg,
        ga_nmax=ga_nmax,
        ga_pop=ga_pop,
        ga_gens=ga_gens,
        ga_elite_frac=ga_elite_frac,
        ga_crossover_prob=ga_crossover_prob,
        ga_mut_sigma_xy=ga_mut_sigma_xy,
        ga_mut_sigma_deg=ga_mut_sigma_deg,
        ga_directed_prob=ga_directed_prob,
        ga_directed_step_xy=ga_directed_step_xy,
        ga_directed_k=ga_directed_k,
        ga_repair_iters=ga_repair_iters,
        ga_hc_passes=ga_hc_passes,
        ga_hc_step_xy=ga_hc_step_xy,
        ga_hc_step_deg=ga_hc_step_deg,
    )
    return base


def _post_optimize(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    n: int,
    seed: int,
    overlap_mode: str,
    refine_nmin: int,
    refine_batch_size: int,
    refine_steps: int,
    refine_trans_sigma: float,
    refine_rot_sigma: float,
    refine_rot_prob: float,
    refine_rot_prob_end: float,
    refine_swap_prob: float,
    refine_swap_prob_end: float,
    refine_push_prob: float,
    refine_push_scale: float,
    refine_push_square_prob: float,
    refine_compact_prob: float,
    refine_compact_prob_end: float,
    refine_compact_scale: float,
    refine_compact_square_prob: float,
    refine_teleport_prob: float,
    refine_teleport_prob_end: float,
    refine_teleport_tries: int,
    refine_teleport_anchor_beta: float,
    refine_teleport_ring_mult: float,
    refine_teleport_jitter: float,
    refine_cooling: str,
    refine_cooling_power: float,
    refine_trans_sigma_nexp: float,
    refine_rot_sigma_nexp: float,
    refine_sigma_nref: float,
    refine_proposal: str,
    refine_smart_prob: float,
    refine_smart_beta: float,
    refine_smart_drift: float,
    refine_smart_noise: float,
    refine_overlap_lambda: float,
    refine_allow_collisions: bool,
    refine_objective: str,
    guided_model: Path | None,
    guided_prob: float,
    guided_pmax: float,
    guided_prob_end: float,
    guided_pmax_end: float,
    lns_nmax: int,
    lns_passes: int,
    lns_destroy_k: int,
    lns_destroy_mode: str,
    lns_tabu_tenure: int,
    lns_candidates: int,
    lns_angle_samples: int,
    lns_pad_scale: float,
    lns_group_moves: int,
    lns_group_size: int,
    lns_group_trans_sigma: float,
    lns_group_rot_sigma: float,
    lns_t_start: float,
    lns_t_end: float,
    hc_nmax: int,
    hc_passes: int,
    hc_step_xy: float,
    hc_step_deg: float,
    ga_nmax: int,
    ga_pop: int,
    ga_gens: int,
    ga_elite_frac: float,
    ga_crossover_prob: float,
    ga_mut_sigma_xy: float,
    ga_mut_sigma_deg: float,
    ga_directed_prob: float,
    ga_directed_step_xy: float,
    ga_directed_k: int,
    ga_repair_iters: int,
    ga_hc_passes: int,
    ga_hc_step_xy: float,
    ga_hc_step_deg: float,
) -> np.ndarray:
    base = np.array(poses, dtype=float, copy=True)

    if refine_steps > 0 and n >= refine_nmin:
        if guided_model is not None:
            refined = _run_sa_guided(
                n,
                model_path=guided_model,
                seed=seed,
                batch_size=refine_batch_size,
                n_steps=refine_steps,
                trans_sigma=refine_trans_sigma,
                rot_sigma=refine_rot_sigma,
                rot_prob=refine_rot_prob,
                rot_prob_end=refine_rot_prob_end,
                swap_prob=refine_swap_prob,
                swap_prob_end=refine_swap_prob_end,
                push_prob=refine_push_prob,
                push_scale=refine_push_scale,
                push_square_prob=refine_push_square_prob,
                compact_prob=refine_compact_prob,
                compact_prob_end=refine_compact_prob_end,
                compact_scale=refine_compact_scale,
                compact_square_prob=refine_compact_square_prob,
                teleport_prob=refine_teleport_prob,
                teleport_prob_end=refine_teleport_prob_end,
                teleport_tries=refine_teleport_tries,
                teleport_anchor_beta=refine_teleport_anchor_beta,
                teleport_ring_mult=refine_teleport_ring_mult,
                teleport_jitter=refine_teleport_jitter,
                cooling=refine_cooling,
                cooling_power=refine_cooling_power,
                trans_sigma_nexp=refine_trans_sigma_nexp,
                rot_sigma_nexp=refine_rot_sigma_nexp,
                sigma_nref=refine_sigma_nref,
                proposal=refine_proposal,
                smart_prob=refine_smart_prob,
                smart_beta=refine_smart_beta,
                smart_drift=refine_smart_drift,
                smart_noise=refine_smart_noise,
                overlap_lambda=refine_overlap_lambda,
                allow_collisions=refine_allow_collisions,
                initial_poses=base,
                objective=refine_objective,
                policy_prob=guided_prob,
                policy_pmax=guided_pmax,
                policy_prob_end=guided_prob_end,
                policy_pmax_end=guided_pmax_end,
            )
        else:
            refined = _run_sa(
                n,
                seed=seed,
                batch_size=refine_batch_size,
                n_steps=refine_steps,
                trans_sigma=refine_trans_sigma,
                rot_sigma=refine_rot_sigma,
                rot_prob=refine_rot_prob,
                rot_prob_end=refine_rot_prob_end,
                swap_prob=refine_swap_prob,
                swap_prob_end=refine_swap_prob_end,
                push_prob=refine_push_prob,
                push_scale=refine_push_scale,
                push_square_prob=refine_push_square_prob,
                compact_prob=refine_compact_prob,
                compact_prob_end=refine_compact_prob_end,
                compact_scale=refine_compact_scale,
                compact_square_prob=refine_compact_square_prob,
                teleport_prob=refine_teleport_prob,
                teleport_prob_end=refine_teleport_prob_end,
                teleport_tries=refine_teleport_tries,
                teleport_anchor_beta=refine_teleport_anchor_beta,
                teleport_ring_mult=refine_teleport_ring_mult,
                teleport_jitter=refine_teleport_jitter,
                cooling=refine_cooling,
                cooling_power=refine_cooling_power,
                trans_sigma_nexp=refine_trans_sigma_nexp,
                rot_sigma_nexp=refine_rot_sigma_nexp,
                sigma_nref=refine_sigma_nref,
                proposal=refine_proposal,
                smart_prob=refine_smart_prob,
                smart_beta=refine_smart_beta,
                smart_drift=refine_smart_drift,
                smart_noise=refine_smart_noise,
                overlap_lambda=refine_overlap_lambda,
                allow_collisions=refine_allow_collisions,
                initial_poses=base,
                objective=refine_objective,
            )
        if refined is not None:
            base = refined

    if lns_passes > 0 and lns_nmax > 0 and n <= lns_nmax:
        from santa_packing.postopt_np import large_neighborhood_search  # noqa: E402

        base = large_neighborhood_search(
            points,
            base,
            seed=seed,
            passes=lns_passes,
            destroy_k=lns_destroy_k,
            destroy_mode=lns_destroy_mode,
            tabu_tenure=lns_tabu_tenure,
            candidates=lns_candidates,
            angle_samples=lns_angle_samples,
            pad_scale=lns_pad_scale,
            group_moves=lns_group_moves,
            group_size=lns_group_size,
            group_trans_sigma=lns_group_trans_sigma,
            group_rot_sigma=lns_group_rot_sigma,
            t_start=lns_t_start,
            t_end=lns_t_end,
            overlap_mode=str(overlap_mode),
        )

    if ga_gens > 0 and ga_nmax > 0 and n <= ga_nmax:
        from santa_packing.postopt_np import genetic_optimize  # noqa: E402

        base = genetic_optimize(
            points,
            [base],
            seed=seed,
            pop_size=ga_pop,
            generations=ga_gens,
            elite_frac=ga_elite_frac,
            crossover_prob=ga_crossover_prob,
            mutation_sigma_xy=ga_mut_sigma_xy,
            mutation_sigma_deg=ga_mut_sigma_deg,
            directed_mut_prob=ga_directed_prob,
            directed_step_xy=ga_directed_step_xy,
            directed_k=ga_directed_k,
            repair_iters=ga_repair_iters,
            hill_climb_passes=ga_hc_passes,
            hill_climb_step_xy=ga_hc_step_xy,
            hill_climb_step_deg=ga_hc_step_deg,
            overlap_mode=str(overlap_mode),
        )

    if hc_passes > 0 and hc_nmax > 0 and n <= hc_nmax:
        from santa_packing.postopt_np import hill_climb  # noqa: E402

        base = hill_climb(
            points,
            base,
            step_xy=hc_step_xy,
            step_deg=hc_step_deg,
            max_passes=hc_passes,
            overlap_mode=str(overlap_mode),
        )

    return base


def main(argv: list[str] | None = None) -> int:
    """Generate a `submission.csv` according to flags and optional JSON config."""
    argv = list(sys.argv[1:] if argv is None else argv)

    cfg_default = default_config_path("submit.json")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None, help="Path to JSON/YAML config (optional)")
    pre.add_argument("--no-config", action="store_true", help="Disable loading the default config (if any).")
    pre_args, _ = pre.parse_known_args(argv)

    if pre_args.no_config and pre_args.config is not None:
        raise SystemExit("Use either --config or --no-config, not both.")

    config_path = None if pre_args.no_config else (pre_args.config or cfg_default)
    config_args = (
        config_to_argv(config_path, section_keys=("generate", "generate_submission")) if config_path is not None else []
    )

    ap = argparse.ArgumentParser(description="Generate submission.csv (hybrid SA + lattice)")
    ap.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="JSON/YAML config file with defaults for this script (defaults to configs/submit.json when present).",
    )
    ap.add_argument("--no-config", action="store_true", help="Disable loading the default config (if any).")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"), help="Output CSV path")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--seed", type=int, default=1, help="Base seed for SA")
    ap.add_argument(
        "--from-submission",
        type=Path,
        default=None,
        help="Load poses from an existing submission.csv and finalize them (optionally applying refine/LNS/HC/GA).",
    )
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="kaggle",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate used during finalization/validation (strict/kaggle allow touching; conservative counts touching).",
    )

    ap.add_argument(
        "--mother-prefix",
        action="store_true",
        help="Solve once for N=nmax and emit radial prefixes for n=1..nmax (nested solutions).",
    )
    ap.add_argument(
        "--mother-reorder",
        type=str,
        default="radial",
        choices=["radial", "none"],
        help="Ordering applied to the mother packing before emitting prefixes (radial recommended; none keeps solver order).",
    )

    ap.add_argument("--sa-nmax", type=int, default=50, help="Use SA for n <= this threshold")
    ap.add_argument("--sa-batch", type=int, default=64, help="SA batch size")
    ap.add_argument("--sa-steps", type=int, default=400, help="SA steps per puzzle")
    ap.add_argument("--sa-trans-sigma", type=float, default=0.2, help="SA translation step scale")
    ap.add_argument("--sa-rot-sigma", type=float, default=15.0, help="SA rotation step scale (deg)")
    ap.add_argument("--sa-rot-prob", type=float, default=0.3, help="SA rotation move probability")
    ap.add_argument(
        "--sa-rot-prob-end",
        type=float,
        default=-1.0,
        help="Final SA rotation move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument(
        "--sa-swap-prob", type=float, default=0.0, help="SA swap move probability (useful for objective=prefix)."
    )
    ap.add_argument(
        "--sa-swap-prob-end",
        type=float,
        default=-1.0,
        help="Final SA swap move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument(
        "--sa-neighborhood",
        action="store_true",
        help="Enable neighborhood moves (swap/compact/teleport) with sensible defaults (unless explicitly set).",
    )
    ap.add_argument(
        "--sa-push-prob",
        type=float,
        default=0.1,
        help="SA deterministic push-to-center move probability (translation-only).",
    )
    ap.add_argument("--sa-push-scale", type=float, default=1.0, help="SA push step magnitude multiplier.")
    ap.add_argument("--sa-push-square-prob", type=float, default=0.5, help="SA push: fraction of axis-aligned pushes.")
    ap.add_argument(
        "--sa-compact-prob", type=float, default=0.0, help="SA compact move probability (boundary -> center)."
    )
    ap.add_argument(
        "--sa-compact-prob-end",
        type=float,
        default=-1.0,
        help="Final SA compact move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument("--sa-compact-scale", type=float, default=1.0, help="SA compact step magnitude multiplier.")
    ap.add_argument(
        "--sa-compact-square-prob", type=float, default=0.75, help="SA compact: fraction of axis-aligned moves."
    )
    ap.add_argument(
        "--sa-teleport-prob",
        type=float,
        default=0.0,
        help="SA teleport move probability (boundary -> interior pocket).",
    )
    ap.add_argument(
        "--sa-teleport-prob-end",
        type=float,
        default=-1.0,
        help="Final SA teleport move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument("--sa-teleport-tries", type=int, default=4, help="SA teleport: number of candidate tries per step.")
    ap.add_argument(
        "--sa-teleport-anchor-beta",
        type=float,
        default=6.0,
        help="SA teleport: bias toward center anchors (higher=more central).",
    )
    ap.add_argument(
        "--sa-teleport-ring-mult", type=float, default=1.02, help="SA teleport: radius multiplier around anchor."
    )
    ap.add_argument(
        "--sa-teleport-jitter", type=float, default=0.05, help="SA teleport: random XY jitter (in radius units)."
    )
    ap.add_argument("--sa-cooling", type=str, default="geom", choices=["geom", "linear", "log"])
    ap.add_argument(
        "--sa-cooling-power", type=float, default=1.0, help="Power on anneal fraction (>=1 slows early cooling)."
    )
    ap.add_argument("--sa-trans-nexp", type=float, default=0.0, help="Scale trans_sigma by (n/nref)^nexp.")
    ap.add_argument("--sa-rot-nexp", type=float, default=0.0, help="Scale rot_sigma by (n/nref)^nexp.")
    ap.add_argument("--sa-sigma-nref", type=float, default=50.0, help="Reference n for sigma scaling.")
    ap.add_argument(
        "--sa-objective",
        type=str,
        default=None,
        choices=["packing", "prefix"],
        help="Objective used by SA. Default: packing (independent) or prefix (when --mother-prefix).",
    )
    ap.add_argument(
        "--sa-proposal",
        type=str,
        default="random",
        choices=["random", "bbox_inward", "bbox", "inward", "smart", "mixed"],
        help="SA proposal mode. 'bbox_inward/smart' targets boundary trees; 'mixed' blends with random.",
    )
    ap.add_argument("--sa-smart-prob", type=float, default=1.0, help="For proposal=mixed: probability of smart move.")
    ap.add_argument(
        "--sa-smart-beta", type=float, default=8.0, help="Edge focus strength (higher=more boundary-biased)."
    )
    ap.add_argument("--sa-smart-drift", type=float, default=1.0, help="Inward drift multiplier (translation moves).")
    ap.add_argument("--sa-smart-noise", type=float, default=0.25, help="Noise multiplier for smart inward moves.")
    ap.add_argument(
        "--sa-overlap-lambda", type=float, default=0.0, help="Energy penalty weight for circle overlap (0 disables)."
    )
    ap.add_argument(
        "--sa-allow-collisions", action="store_true", help="Allow accepting colliding states (best kept feasible)."
    )
    ap.add_argument("--meta-init-model", type=Path, default=None, help="Meta-init model (.npz) for SA init")
    ap.add_argument("--heatmap-model", type=Path, default=None, help="Heatmap meta-optimizer model (.npz)")
    ap.add_argument("--heatmap-nmax", type=int, default=10, help="Use heatmap for n <= this threshold")
    ap.add_argument("--heatmap-steps", type=int, default=200, help="Heatmap search steps per puzzle")

    ap.add_argument("--l2o-model", type=Path, default=None, help="Path to L2O policy (.npz)")
    ap.add_argument("--l2o-init", type=str, default="grid", choices=["grid", "lattice"], help="Initial poses for L2O")
    ap.add_argument("--l2o-nmax", type=int, default=10, help="Use L2O for n <= this threshold")
    ap.add_argument("--l2o-steps", type=int, default=200, help="L2O rollout steps per puzzle")
    ap.add_argument("--l2o-trans-sigma", type=float, default=0.2, help="L2O translation step scale")
    ap.add_argument("--l2o-rot-sigma", type=float, default=10.0, help="L2O rotation step scale")
    ap.add_argument("--l2o-deterministic", action="store_true", help="Disable L2O action noise")

    ap.add_argument("--lattice-pattern", type=str, default="hex", choices=["hex", "square"])
    ap.add_argument("--lattice-margin", type=float, default=0.02, help="Relative spacing margin")
    ap.add_argument("--lattice-rotate", type=float, default=0.0, help="Constant rotation (deg)")
    ap.add_argument(
        "--lattice-rotate-mode",
        type=str,
        default="constant",
        choices=["constant", "row", "checker", "ring"],
        help="Rotation pattern for lattice. If not constant, uses --lattice-rotations as a repeating sequence.",
    )
    ap.add_argument(
        "--lattice-rotations",
        type=str,
        default="0,15,30",
        help="Comma-separated rotations (deg). In constant mode: try each value and pick best per n; otherwise: repeating sequence.",
    )
    ap.add_argument(
        "--lattice-post-nmax",
        type=int,
        default=0,
        help="Apply a short boundary hill-climb right after lattice for n <= this threshold (0=disabled).",
    )
    ap.add_argument("--lattice-post-steps", type=int, default=50, help="Post-lattice boundary steps.")
    ap.add_argument("--lattice-post-step-xy", type=float, default=0.01, help="Post-lattice translation step.")
    ap.add_argument("--lattice-post-step-deg", type=float, default=0.0, help="Post-lattice rotation step (deg).")

    ap.add_argument(
        "--refine-nmin", type=int, default=0, help="Refine lattice with SA for n >= this threshold (0=disabled)"
    )
    ap.add_argument("--refine-batch", type=int, default=16, help="Refine SA batch size")
    ap.add_argument("--refine-steps", type=int, default=0, help="Refine SA steps per puzzle (0=disabled)")
    ap.add_argument("--refine-trans-sigma", type=float, default=0.2, help="Refine SA translation step scale")
    ap.add_argument("--refine-rot-sigma", type=float, default=15.0, help="Refine SA rotation step scale (deg)")
    ap.add_argument("--refine-rot-prob", type=float, default=0.3, help="Refine SA rotation move probability")
    ap.add_argument(
        "--refine-rot-prob-end",
        type=float,
        default=-1.0,
        help="Final refine rotation move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument(
        "--refine-swap-prob",
        type=float,
        default=0.0,
        help="Refine SA swap move probability (useful for objective=prefix).",
    )
    ap.add_argument(
        "--refine-swap-prob-end",
        type=float,
        default=-1.0,
        help="Final refine swap move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument(
        "--refine-neighborhood",
        action="store_true",
        help="Enable neighborhood moves (swap/compact/teleport) with sensible defaults (unless explicitly set).",
    )
    ap.add_argument(
        "--refine-push-prob",
        type=float,
        default=0.1,
        help="Refine SA deterministic push-to-center move probability (translation-only).",
    )
    ap.add_argument("--refine-push-scale", type=float, default=1.0, help="Refine SA push step magnitude multiplier.")
    ap.add_argument(
        "--refine-push-square-prob", type=float, default=0.5, help="Refine SA push: fraction of axis-aligned pushes."
    )
    ap.add_argument(
        "--refine-compact-prob",
        type=float,
        default=0.0,
        help="Refine SA compact move probability (boundary -> center).",
    )
    ap.add_argument(
        "--refine-compact-prob-end",
        type=float,
        default=-1.0,
        help="Final refine compact move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument(
        "--refine-compact-scale", type=float, default=1.0, help="Refine SA compact step magnitude multiplier."
    )
    ap.add_argument(
        "--refine-compact-square-prob",
        type=float,
        default=0.75,
        help="Refine SA compact: fraction of axis-aligned moves.",
    )
    ap.add_argument(
        "--refine-teleport-prob",
        type=float,
        default=0.0,
        help="Refine SA teleport move probability (boundary -> interior pocket).",
    )
    ap.add_argument(
        "--refine-teleport-prob-end",
        type=float,
        default=-1.0,
        help="Final refine teleport move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument(
        "--refine-teleport-tries", type=int, default=4, help="Refine SA teleport: number of candidate tries per step."
    )
    ap.add_argument(
        "--refine-teleport-anchor-beta",
        type=float,
        default=6.0,
        help="Refine SA teleport: bias toward center anchors (higher=more central).",
    )
    ap.add_argument(
        "--refine-teleport-ring-mult",
        type=float,
        default=1.02,
        help="Refine SA teleport: radius multiplier around anchor.",
    )
    ap.add_argument(
        "--refine-teleport-jitter",
        type=float,
        default=0.05,
        help="Refine SA teleport: random XY jitter (in radius units).",
    )
    ap.add_argument("--refine-cooling", type=str, default="geom", choices=["geom", "linear", "log"])
    ap.add_argument("--refine-cooling-power", type=float, default=1.0)
    ap.add_argument("--refine-trans-nexp", type=float, default=0.0)
    ap.add_argument("--refine-rot-nexp", type=float, default=0.0)
    ap.add_argument("--refine-sigma-nref", type=float, default=50.0)
    ap.add_argument(
        "--refine-objective",
        type=str,
        default=None,
        choices=["packing", "prefix"],
        help="Objective used by refine-SA. Default: matches --sa-objective (after auto selection).",
    )
    ap.add_argument(
        "--refine-proposal",
        type=str,
        default="random",
        choices=["random", "bbox_inward", "bbox", "inward", "smart", "mixed"],
        help="Refine SA proposal mode. 'bbox_inward/smart' targets boundary trees; 'mixed' blends with random.",
    )
    ap.add_argument(
        "--refine-smart-prob", type=float, default=1.0, help="For proposal=mixed: probability of smart move."
    )
    ap.add_argument(
        "--refine-smart-beta", type=float, default=8.0, help="Edge focus strength (higher=more boundary-biased)."
    )
    ap.add_argument(
        "--refine-smart-drift", type=float, default=1.0, help="Inward drift multiplier (translation moves)."
    )
    ap.add_argument("--refine-smart-noise", type=float, default=0.25, help="Noise multiplier for smart inward moves.")
    ap.add_argument(
        "--refine-overlap-lambda",
        type=float,
        default=0.0,
        help="Energy penalty weight for circle overlap (0 disables).",
    )
    ap.add_argument(
        "--refine-allow-collisions", action="store_true", help="Allow accepting colliding states (best kept feasible)."
    )

    ap.add_argument("--guided-model", type=Path, default=None, help="L2O policy (.npz) used as SA proposal generator")
    ap.add_argument(
        "--guided-prob", type=float, default=1.0, help="Probability of using policy proposal (when confident)"
    )
    ap.add_argument(
        "--guided-pmax", type=float, default=0.05, help="Min max-softmax(logits) to consider policy confident"
    )
    ap.add_argument(
        "--guided-prob-end",
        type=float,
        default=-1.0,
        help="Final probability of using policy proposal (linear schedule; -1 keeps constant).",
    )
    ap.add_argument(
        "--guided-pmax-end",
        type=float,
        default=-1.0,
        help="Final policy confidence threshold (linear schedule; -1 keeps constant).",
    )

    ap.add_argument("--block-nmax", type=int, default=0, help="Apply block SA for n <= this threshold (0=disabled).")
    ap.add_argument("--block-size", type=int, default=2, help="Block size (trees per block). Typical: 2..4.")
    ap.add_argument("--block-batch", type=int, default=32, help="Block SA batch size.")
    ap.add_argument("--block-steps", type=int, default=0, help="Block SA steps per puzzle (0=disabled).")
    ap.add_argument("--block-trans-sigma", type=float, default=0.2, help="Block SA translation step scale.")
    ap.add_argument("--block-rot-sigma", type=float, default=15.0, help="Block SA rotation step scale (deg).")
    ap.add_argument("--block-rot-prob", type=float, default=0.25, help="Block SA rotation move probability.")
    ap.add_argument(
        "--block-rot-prob-end",
        type=float,
        default=-1.0,
        help="Final block rotation move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument("--block-cooling", type=str, default="geom", choices=["geom", "linear", "log"])
    ap.add_argument("--block-cooling-power", type=float, default=1.0)
    ap.add_argument("--block-trans-nexp", type=float, default=0.0)
    ap.add_argument("--block-rot-nexp", type=float, default=0.0)
    ap.add_argument("--block-sigma-nref", type=float, default=50.0)
    ap.add_argument("--block-overlap-lambda", type=float, default=0.0)
    ap.add_argument("--block-allow-collisions", action="store_true")
    ap.add_argument(
        "--block-objective",
        type=str,
        default=None,
        choices=["packing", "prefix"],
        help="Objective used by block-SA. Default: matches --sa-objective (after auto selection).",
    )
    ap.add_argument(
        "--block-init",
        type=str,
        default="cluster",
        choices=["cluster", "template"],
        help="Block init mode: cluster blocks from an existing packing, or generate from a template.",
    )
    ap.add_argument("--block-template-pattern", type=str, default="hex", choices=["hex", "square"])
    ap.add_argument("--block-template-margin", type=float, default=0.02)
    ap.add_argument("--block-template-rotate", type=float, default=0.0)

    ap.add_argument(
        "--lns-nmax", type=int, default=0, help="Apply LNS/ALNS post-optimization for n <= this threshold (0=disabled)."
    )
    ap.add_argument("--lns-passes", type=int, default=0, help="LNS passes (0=disabled).")
    ap.add_argument("--lns-destroy-k", type=int, default=8, help="Trees removed per ruin&recreate pass.")
    ap.add_argument(
        "--lns-destroy-mode", type=str, default="mixed", choices=["mixed", "boundary", "random", "cluster", "alns"]
    )
    ap.add_argument(
        "--lns-tabu-tenure", type=int, default=0, help="Tabu tenure (passes) for recent ruin-sets (0 disables)."
    )
    ap.add_argument("--lns-candidates", type=int, default=64, help="Candidate centers per reinsertion (sampling).")
    ap.add_argument("--lns-angle-samples", type=int, default=8, help="Angle samples per candidate reinsertion.")
    ap.add_argument(
        "--lns-pad-scale",
        type=float,
        default=2.0,
        help="Sampling padding around current bbox in multiples of 2*radius.",
    )
    ap.add_argument("--lns-group-moves", type=int, default=0, help="Group-rotation proposals per pass (0 disables).")
    ap.add_argument("--lns-group-size", type=int, default=3, help="Group size for group rotations.")
    ap.add_argument(
        "--lns-group-trans-sigma", type=float, default=0.05, help="Group translation sigma (same units as x/y)."
    )
    ap.add_argument("--lns-group-rot-sigma", type=float, default=20.0, help="Group rotation sigma (deg).")
    ap.add_argument(
        "--lns-t-start", type=float, default=0.0, help="If >0, enable SA acceptance in LNS: start temperature."
    )
    ap.add_argument("--lns-t-end", type=float, default=0.0, help="End temperature for LNS SA acceptance.")

    ap.add_argument(
        "--hc-nmax", type=int, default=0, help="Apply deterministic hill-climb for n <= this threshold (0=disabled)."
    )
    ap.add_argument("--hc-passes", type=int, default=2, help="Hill-climb passes over trees.")
    ap.add_argument("--hc-step-xy", type=float, default=0.01, help="Hill-climb translation step.")
    ap.add_argument("--hc-step-deg", type=float, default=2.0, help="Hill-climb rotation step (deg).")

    ap.add_argument("--ga-nmax", type=int, default=0, help="Apply GA refinement for n <= this threshold (0=disabled).")
    ap.add_argument("--ga-pop", type=int, default=24, help="GA population size.")
    ap.add_argument("--ga-gens", type=int, default=20, help="GA generations.")
    ap.add_argument("--ga-elite-frac", type=float, default=0.25, help="Elite fraction carried over each generation.")
    ap.add_argument("--ga-crossover-prob", type=float, default=0.5, help="Crossover probability.")
    ap.add_argument("--ga-mut-sigma-xy", type=float, default=0.01, help="Mutation translation sigma.")
    ap.add_argument("--ga-mut-sigma-deg", type=float, default=2.0, help="Mutation rotation sigma (deg).")
    ap.add_argument(
        "--ga-directed-prob", type=float, default=0.5, help="Probability of using directed (bbox-inward) mutation."
    )
    ap.add_argument("--ga-directed-step-xy", type=float, default=0.02, help="Directed mutation drift step.")
    ap.add_argument(
        "--ga-directed-k", type=int, default=8, help="Directed mutation samples from the k most boundary-ish trees."
    )
    ap.add_argument("--ga-repair-iters", type=int, default=200, help="Max repair iterations for colliding children.")
    ap.add_argument(
        "--ga-hc-passes", type=int, default=0, help="Optional hill-climb passes applied inside GA (0=disabled)."
    )
    ap.add_argument("--ga-hc-step-xy", type=float, default=0.01)
    ap.add_argument("--ga-hc-step-deg", type=float, default=2.0)
    args = ap.parse_args(config_args + argv)
    lattice_rotate_degs = _parse_float_list(args.lattice_rotations)
    if args.sa_objective is None:
        args.sa_objective = "prefix" if args.mother_prefix else "packing"
    if args.refine_objective is None:
        args.refine_objective = args.sa_objective
    if args.block_objective is None:
        args.block_objective = args.sa_objective

    if args.sa_neighborhood:
        if args.sa_proposal == "random":
            args.sa_proposal = "mixed"
        if args.sa_objective == "prefix" and args.sa_swap_prob == 0.0:
            args.sa_swap_prob = 0.2
        if args.sa_compact_prob == 0.0:
            args.sa_compact_prob = 0.1
        if args.sa_teleport_prob == 0.0:
            args.sa_teleport_prob = 0.03

    if args.refine_neighborhood:
        if args.refine_proposal == "random":
            args.refine_proposal = "mixed"
        if args.refine_objective == "prefix" and args.refine_swap_prob == 0.0:
            args.refine_swap_prob = 0.2
        if args.refine_compact_prob == 0.0:
            args.refine_compact_prob = 0.1
        if args.refine_teleport_prob == 0.0:
            args.refine_teleport_prob = 0.03

    points = np.array(TREE_POINTS, dtype=float)
    puzzles: dict[int, np.ndarray] = {}

    if args.from_submission is not None:
        from santa_packing.scoring import load_submission  # noqa: E402

        loaded = load_submission(args.from_submission, nmax=int(args.nmax))
        postopt_enabled = bool(
            (args.refine_steps > 0)
            or (args.lns_passes > 0 and args.lns_nmax > 0)
            or (args.hc_passes > 0 and args.hc_nmax > 0)
            or (args.ga_gens > 0 and args.ga_nmax > 0)
        )
        for n in range(1, args.nmax + 1):
            poses = loaded.get(n)
            if poses is None or poses.shape != (n, 3):
                raise SystemExit(
                    f"--from-submission: missing puzzle {n} or wrong shape (got {None if poses is None else poses.shape})"
                )
            if not postopt_enabled:
                puzzles[n] = _finalize_puzzle(
                    points,
                    poses,
                    seed=args.seed + n,
                    puzzle_n=n,
                    overlap_mode=str(args.overlap_mode),
                )
                continue

            base = _finalize_puzzle(
                points,
                poses,
                seed=args.seed + n,
                puzzle_n=n,
                overlap_mode=str(args.overlap_mode),
            )
            base = _post_optimize(
                points,
                base,
                n=n,
                seed=args.seed + n,
                overlap_mode=str(args.overlap_mode),
                refine_nmin=args.refine_nmin,
                refine_batch_size=args.refine_batch,
                refine_steps=args.refine_steps,
                refine_trans_sigma=args.refine_trans_sigma,
                refine_rot_sigma=args.refine_rot_sigma,
                refine_rot_prob=args.refine_rot_prob,
                refine_rot_prob_end=args.refine_rot_prob_end,
                refine_swap_prob=args.refine_swap_prob,
                refine_swap_prob_end=args.refine_swap_prob_end,
                refine_push_prob=args.refine_push_prob,
                refine_push_scale=args.refine_push_scale,
                refine_push_square_prob=args.refine_push_square_prob,
                refine_compact_prob=args.refine_compact_prob,
                refine_compact_prob_end=args.refine_compact_prob_end,
                refine_compact_scale=args.refine_compact_scale,
                refine_compact_square_prob=args.refine_compact_square_prob,
                refine_teleport_prob=args.refine_teleport_prob,
                refine_teleport_prob_end=args.refine_teleport_prob_end,
                refine_teleport_tries=args.refine_teleport_tries,
                refine_teleport_anchor_beta=args.refine_teleport_anchor_beta,
                refine_teleport_ring_mult=args.refine_teleport_ring_mult,
                refine_teleport_jitter=args.refine_teleport_jitter,
                refine_cooling=args.refine_cooling,
                refine_cooling_power=args.refine_cooling_power,
                refine_trans_sigma_nexp=args.refine_trans_nexp,
                refine_rot_sigma_nexp=args.refine_rot_nexp,
                refine_sigma_nref=args.refine_sigma_nref,
                refine_proposal=args.refine_proposal,
                refine_smart_prob=args.refine_smart_prob,
                refine_smart_beta=args.refine_smart_beta,
                refine_smart_drift=args.refine_smart_drift,
                refine_smart_noise=args.refine_smart_noise,
                refine_overlap_lambda=args.refine_overlap_lambda,
                refine_allow_collisions=args.refine_allow_collisions,
                refine_objective=str(args.refine_objective),
                guided_model=args.guided_model,
                guided_prob=args.guided_prob,
                guided_pmax=args.guided_pmax,
                guided_prob_end=args.guided_prob_end,
                guided_pmax_end=args.guided_pmax_end,
                lns_nmax=args.lns_nmax,
                lns_passes=args.lns_passes,
                lns_destroy_k=args.lns_destroy_k,
                lns_destroy_mode=args.lns_destroy_mode,
                lns_tabu_tenure=args.lns_tabu_tenure,
                lns_candidates=args.lns_candidates,
                lns_angle_samples=args.lns_angle_samples,
                lns_pad_scale=args.lns_pad_scale,
                lns_group_moves=args.lns_group_moves,
                lns_group_size=args.lns_group_size,
                lns_group_trans_sigma=args.lns_group_trans_sigma,
                lns_group_rot_sigma=args.lns_group_rot_sigma,
                lns_t_start=args.lns_t_start,
                lns_t_end=args.lns_t_end,
                hc_nmax=args.hc_nmax,
                hc_passes=args.hc_passes,
                hc_step_xy=args.hc_step_xy,
                hc_step_deg=args.hc_step_deg,
                ga_nmax=args.ga_nmax,
                ga_pop=args.ga_pop,
                ga_gens=args.ga_gens,
                ga_elite_frac=args.ga_elite_frac,
                ga_crossover_prob=args.ga_crossover_prob,
                ga_mut_sigma_xy=args.ga_mut_sigma_xy,
                ga_mut_sigma_deg=args.ga_mut_sigma_deg,
                ga_directed_prob=args.ga_directed_prob,
                ga_directed_step_xy=args.ga_directed_step_xy,
                ga_directed_k=args.ga_directed_k,
                ga_repair_iters=args.ga_repair_iters,
                ga_hc_passes=args.ga_hc_passes,
                ga_hc_step_xy=args.ga_hc_step_xy,
                ga_hc_step_deg=args.ga_hc_step_deg,
            )
            puzzles[n] = _finalize_puzzle(
                points,
                base,
                seed=args.seed + n + 13_000,
                puzzle_n=n,
                overlap_mode=str(args.overlap_mode),
            )
    elif args.mother_prefix:
        mother = solve_n(
            args.nmax,
            seed=args.seed + args.nmax,
            overlap_mode=str(args.overlap_mode),
            lattice_pattern=args.lattice_pattern,
            lattice_margin=args.lattice_margin,
            lattice_rotate_deg=args.lattice_rotate,
            lattice_rotate_mode=args.lattice_rotate_mode,
            lattice_rotate_degs=lattice_rotate_degs,
            lattice_post_nmax=args.lattice_post_nmax,
            lattice_post_steps=args.lattice_post_steps,
            lattice_post_step_xy=args.lattice_post_step_xy,
            lattice_post_step_deg=args.lattice_post_step_deg,
            sa_nmax=args.sa_nmax,
            sa_batch_size=args.sa_batch,
            sa_steps=args.sa_steps,
            sa_trans_sigma=args.sa_trans_sigma,
            sa_rot_sigma=args.sa_rot_sigma,
            sa_rot_prob=args.sa_rot_prob,
            sa_rot_prob_end=args.sa_rot_prob_end,
            sa_swap_prob=args.sa_swap_prob,
            sa_swap_prob_end=args.sa_swap_prob_end,
            sa_push_prob=args.sa_push_prob,
            sa_push_scale=args.sa_push_scale,
            sa_push_square_prob=args.sa_push_square_prob,
            sa_compact_prob=args.sa_compact_prob,
            sa_compact_prob_end=args.sa_compact_prob_end,
            sa_compact_scale=args.sa_compact_scale,
            sa_compact_square_prob=args.sa_compact_square_prob,
            sa_teleport_prob=args.sa_teleport_prob,
            sa_teleport_prob_end=args.sa_teleport_prob_end,
            sa_teleport_tries=args.sa_teleport_tries,
            sa_teleport_anchor_beta=args.sa_teleport_anchor_beta,
            sa_teleport_ring_mult=args.sa_teleport_ring_mult,
            sa_teleport_jitter=args.sa_teleport_jitter,
            sa_cooling=args.sa_cooling,
            sa_cooling_power=args.sa_cooling_power,
            sa_trans_sigma_nexp=args.sa_trans_nexp,
            sa_rot_sigma_nexp=args.sa_rot_nexp,
            sa_sigma_nref=args.sa_sigma_nref,
            sa_proposal=args.sa_proposal,
            sa_smart_prob=args.sa_smart_prob,
            sa_smart_beta=args.sa_smart_beta,
            sa_smart_drift=args.sa_smart_drift,
            sa_smart_noise=args.sa_smart_noise,
            sa_overlap_lambda=args.sa_overlap_lambda,
            sa_allow_collisions=args.sa_allow_collisions,
            sa_objective=args.sa_objective,
            meta_init_model=args.meta_init_model,
            heatmap_model=args.heatmap_model,
            heatmap_nmax=args.heatmap_nmax,
            heatmap_steps=args.heatmap_steps,
            l2o_model=args.l2o_model,
            l2o_init=args.l2o_init,
            l2o_nmax=args.l2o_nmax,
            l2o_steps=args.l2o_steps,
            l2o_trans_sigma=args.l2o_trans_sigma,
            l2o_rot_sigma=args.l2o_rot_sigma,
            l2o_deterministic=args.l2o_deterministic,
            refine_nmin=args.refine_nmin,
            refine_batch_size=args.refine_batch,
            refine_steps=args.refine_steps,
            refine_trans_sigma=args.refine_trans_sigma,
            refine_rot_sigma=args.refine_rot_sigma,
            refine_rot_prob=args.refine_rot_prob,
            refine_rot_prob_end=args.refine_rot_prob_end,
            refine_swap_prob=args.refine_swap_prob,
            refine_swap_prob_end=args.refine_swap_prob_end,
            refine_push_prob=args.refine_push_prob,
            refine_push_scale=args.refine_push_scale,
            refine_push_square_prob=args.refine_push_square_prob,
            refine_compact_prob=args.refine_compact_prob,
            refine_compact_prob_end=args.refine_compact_prob_end,
            refine_compact_scale=args.refine_compact_scale,
            refine_compact_square_prob=args.refine_compact_square_prob,
            refine_teleport_prob=args.refine_teleport_prob,
            refine_teleport_prob_end=args.refine_teleport_prob_end,
            refine_teleport_tries=args.refine_teleport_tries,
            refine_teleport_anchor_beta=args.refine_teleport_anchor_beta,
            refine_teleport_ring_mult=args.refine_teleport_ring_mult,
            refine_teleport_jitter=args.refine_teleport_jitter,
            refine_cooling=args.refine_cooling,
            refine_cooling_power=args.refine_cooling_power,
            refine_trans_sigma_nexp=args.refine_trans_nexp,
            refine_rot_sigma_nexp=args.refine_rot_nexp,
            refine_sigma_nref=args.refine_sigma_nref,
            refine_proposal=args.refine_proposal,
            refine_smart_prob=args.refine_smart_prob,
            refine_smart_beta=args.refine_smart_beta,
            refine_smart_drift=args.refine_smart_drift,
            refine_smart_noise=args.refine_smart_noise,
            refine_overlap_lambda=args.refine_overlap_lambda,
            refine_allow_collisions=args.refine_allow_collisions,
            refine_objective=args.refine_objective,
            lns_nmax=args.lns_nmax,
            lns_passes=args.lns_passes,
            lns_destroy_k=args.lns_destroy_k,
            lns_destroy_mode=args.lns_destroy_mode,
            lns_tabu_tenure=args.lns_tabu_tenure,
            lns_candidates=args.lns_candidates,
            lns_angle_samples=args.lns_angle_samples,
            lns_pad_scale=args.lns_pad_scale,
            lns_group_moves=args.lns_group_moves,
            lns_group_size=args.lns_group_size,
            lns_group_trans_sigma=args.lns_group_trans_sigma,
            lns_group_rot_sigma=args.lns_group_rot_sigma,
            lns_t_start=args.lns_t_start,
            lns_t_end=args.lns_t_end,
            hc_nmax=args.hc_nmax,
            hc_passes=args.hc_passes,
            hc_step_xy=args.hc_step_xy,
            hc_step_deg=args.hc_step_deg,
            ga_nmax=args.ga_nmax,
            ga_pop=args.ga_pop,
            ga_gens=args.ga_gens,
            ga_elite_frac=args.ga_elite_frac,
            ga_crossover_prob=args.ga_crossover_prob,
            ga_mut_sigma_xy=args.ga_mut_sigma_xy,
            ga_mut_sigma_deg=args.ga_mut_sigma_deg,
            ga_directed_prob=args.ga_directed_prob,
            ga_directed_step_xy=args.ga_directed_step_xy,
            ga_directed_k=args.ga_directed_k,
            ga_repair_iters=args.ga_repair_iters,
            ga_hc_passes=args.ga_hc_passes,
            ga_hc_step_xy=args.ga_hc_step_xy,
            ga_hc_step_deg=args.ga_hc_step_deg,
            guided_model=args.guided_model,
            guided_prob=args.guided_prob,
            guided_pmax=args.guided_pmax,
            guided_prob_end=args.guided_prob_end,
            guided_pmax_end=args.guided_pmax_end,
            block_nmax=args.block_nmax,
            block_size=args.block_size,
            block_batch_size=args.block_batch,
            block_steps=args.block_steps,
            block_trans_sigma=args.block_trans_sigma,
            block_rot_sigma=args.block_rot_sigma,
            block_rot_prob=args.block_rot_prob,
            block_rot_prob_end=args.block_rot_prob_end,
            block_cooling=args.block_cooling,
            block_cooling_power=args.block_cooling_power,
            block_trans_sigma_nexp=args.block_trans_nexp,
            block_rot_sigma_nexp=args.block_rot_nexp,
            block_sigma_nref=args.block_sigma_nref,
            block_overlap_lambda=args.block_overlap_lambda,
            block_allow_collisions=args.block_allow_collisions,
            block_objective=args.block_objective,
            block_init=args.block_init,
            block_template_pattern=args.block_template_pattern,
            block_template_margin=args.block_template_margin,
            block_template_rotate=args.block_template_rotate,
        )

        mother = np.array(mother, dtype=float)
        mother[:, 2] = np.mod(mother[:, 2], 360.0)
        if args.mother_reorder == "radial":
            mother = _radial_reorder(points, mother)

        for n in range(1, args.nmax + 1):
            poses = shift_poses_to_origin(points, mother[:n])
            puzzles[n] = _finalize_puzzle(
                points, poses, seed=args.seed + n, puzzle_n=n, overlap_mode=str(args.overlap_mode)
            )
    else:
        for n in range(1, args.nmax + 1):
            poses = solve_n(
                n,
                seed=args.seed + n,
                overlap_mode=str(args.overlap_mode),
                lattice_pattern=args.lattice_pattern,
                lattice_margin=args.lattice_margin,
                lattice_rotate_deg=args.lattice_rotate,
                lattice_rotate_mode=args.lattice_rotate_mode,
                lattice_rotate_degs=lattice_rotate_degs,
                lattice_post_nmax=args.lattice_post_nmax,
                lattice_post_steps=args.lattice_post_steps,
                lattice_post_step_xy=args.lattice_post_step_xy,
                lattice_post_step_deg=args.lattice_post_step_deg,
                sa_nmax=args.sa_nmax,
                sa_batch_size=args.sa_batch,
                sa_steps=args.sa_steps,
                sa_trans_sigma=args.sa_trans_sigma,
                sa_rot_sigma=args.sa_rot_sigma,
                sa_rot_prob=args.sa_rot_prob,
                sa_rot_prob_end=args.sa_rot_prob_end,
                sa_swap_prob=args.sa_swap_prob,
                sa_swap_prob_end=args.sa_swap_prob_end,
                sa_push_prob=args.sa_push_prob,
                sa_push_scale=args.sa_push_scale,
                sa_push_square_prob=args.sa_push_square_prob,
                sa_compact_prob=args.sa_compact_prob,
                sa_compact_prob_end=args.sa_compact_prob_end,
                sa_compact_scale=args.sa_compact_scale,
                sa_compact_square_prob=args.sa_compact_square_prob,
                sa_teleport_prob=args.sa_teleport_prob,
                sa_teleport_prob_end=args.sa_teleport_prob_end,
                sa_teleport_tries=args.sa_teleport_tries,
                sa_teleport_anchor_beta=args.sa_teleport_anchor_beta,
                sa_teleport_ring_mult=args.sa_teleport_ring_mult,
                sa_teleport_jitter=args.sa_teleport_jitter,
                sa_cooling=args.sa_cooling,
                sa_cooling_power=args.sa_cooling_power,
                sa_trans_sigma_nexp=args.sa_trans_nexp,
                sa_rot_sigma_nexp=args.sa_rot_nexp,
                sa_sigma_nref=args.sa_sigma_nref,
                sa_proposal=args.sa_proposal,
                sa_smart_prob=args.sa_smart_prob,
                sa_smart_beta=args.sa_smart_beta,
                sa_smart_drift=args.sa_smart_drift,
                sa_smart_noise=args.sa_smart_noise,
                sa_overlap_lambda=args.sa_overlap_lambda,
                sa_allow_collisions=args.sa_allow_collisions,
                sa_objective=args.sa_objective,
                meta_init_model=args.meta_init_model,
                heatmap_model=args.heatmap_model,
                heatmap_nmax=args.heatmap_nmax,
                heatmap_steps=args.heatmap_steps,
                l2o_model=args.l2o_model,
                l2o_init=args.l2o_init,
                l2o_nmax=args.l2o_nmax,
                l2o_steps=args.l2o_steps,
                l2o_trans_sigma=args.l2o_trans_sigma,
                l2o_rot_sigma=args.l2o_rot_sigma,
                l2o_deterministic=args.l2o_deterministic,
                refine_nmin=args.refine_nmin,
                refine_batch_size=args.refine_batch,
                refine_steps=args.refine_steps,
                refine_trans_sigma=args.refine_trans_sigma,
                refine_rot_sigma=args.refine_rot_sigma,
                refine_rot_prob=args.refine_rot_prob,
                refine_rot_prob_end=args.refine_rot_prob_end,
                refine_swap_prob=args.refine_swap_prob,
                refine_swap_prob_end=args.refine_swap_prob_end,
                refine_push_prob=args.refine_push_prob,
                refine_push_scale=args.refine_push_scale,
                refine_push_square_prob=args.refine_push_square_prob,
                refine_compact_prob=args.refine_compact_prob,
                refine_compact_prob_end=args.refine_compact_prob_end,
                refine_compact_scale=args.refine_compact_scale,
                refine_compact_square_prob=args.refine_compact_square_prob,
                refine_teleport_prob=args.refine_teleport_prob,
                refine_teleport_prob_end=args.refine_teleport_prob_end,
                refine_teleport_tries=args.refine_teleport_tries,
                refine_teleport_anchor_beta=args.refine_teleport_anchor_beta,
                refine_teleport_ring_mult=args.refine_teleport_ring_mult,
                refine_teleport_jitter=args.refine_teleport_jitter,
                refine_cooling=args.refine_cooling,
                refine_cooling_power=args.refine_cooling_power,
                refine_trans_sigma_nexp=args.refine_trans_nexp,
                refine_rot_sigma_nexp=args.refine_rot_nexp,
                refine_sigma_nref=args.refine_sigma_nref,
                refine_proposal=args.refine_proposal,
                refine_smart_prob=args.refine_smart_prob,
                refine_smart_beta=args.refine_smart_beta,
                refine_smart_drift=args.refine_smart_drift,
                refine_smart_noise=args.refine_smart_noise,
                refine_overlap_lambda=args.refine_overlap_lambda,
                refine_allow_collisions=args.refine_allow_collisions,
                refine_objective=args.refine_objective,
                lns_nmax=args.lns_nmax,
                lns_passes=args.lns_passes,
                lns_destroy_k=args.lns_destroy_k,
                lns_destroy_mode=args.lns_destroy_mode,
                lns_tabu_tenure=args.lns_tabu_tenure,
                lns_candidates=args.lns_candidates,
                lns_angle_samples=args.lns_angle_samples,
                lns_pad_scale=args.lns_pad_scale,
                lns_group_moves=args.lns_group_moves,
                lns_group_size=args.lns_group_size,
                lns_group_trans_sigma=args.lns_group_trans_sigma,
                lns_group_rot_sigma=args.lns_group_rot_sigma,
                lns_t_start=args.lns_t_start,
                lns_t_end=args.lns_t_end,
                hc_nmax=args.hc_nmax,
                hc_passes=args.hc_passes,
                hc_step_xy=args.hc_step_xy,
                hc_step_deg=args.hc_step_deg,
                ga_nmax=args.ga_nmax,
                ga_pop=args.ga_pop,
                ga_gens=args.ga_gens,
                ga_elite_frac=args.ga_elite_frac,
                ga_crossover_prob=args.ga_crossover_prob,
                ga_mut_sigma_xy=args.ga_mut_sigma_xy,
                ga_mut_sigma_deg=args.ga_mut_sigma_deg,
                ga_directed_prob=args.ga_directed_prob,
                ga_directed_step_xy=args.ga_directed_step_xy,
                ga_directed_k=args.ga_directed_k,
                ga_repair_iters=args.ga_repair_iters,
                ga_hc_passes=args.ga_hc_passes,
                ga_hc_step_xy=args.ga_hc_step_xy,
                ga_hc_step_deg=args.ga_hc_step_deg,
                guided_model=args.guided_model,
                guided_prob=args.guided_prob,
                guided_pmax=args.guided_pmax,
                guided_prob_end=args.guided_prob_end,
                guided_pmax_end=args.guided_pmax_end,
                block_nmax=args.block_nmax,
                block_size=args.block_size,
                block_batch_size=args.block_batch,
                block_steps=args.block_steps,
                block_trans_sigma=args.block_trans_sigma,
                block_rot_sigma=args.block_rot_sigma,
                block_rot_prob=args.block_rot_prob,
                block_rot_prob_end=args.block_rot_prob_end,
                block_cooling=args.block_cooling,
                block_cooling_power=args.block_cooling_power,
                block_trans_sigma_nexp=args.block_trans_nexp,
                block_rot_sigma_nexp=args.block_rot_nexp,
                block_sigma_nref=args.block_sigma_nref,
                block_overlap_lambda=args.block_overlap_lambda,
                block_allow_collisions=args.block_allow_collisions,
                block_objective=args.block_objective,
                block_init=args.block_init,
                block_template_pattern=args.block_template_pattern,
                block_template_margin=args.block_template_margin,
                block_template_rotate=args.block_template_rotate,
            )

            puzzles[n] = _finalize_puzzle(
                points, poses, seed=args.seed + n, puzzle_n=n, overlap_mode=str(args.overlap_mode)
            )

    # Write CSV (after overlap validation/repair).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])
        for n in range(1, args.nmax + 1):
            poses = puzzles[n]
            for i, (x, y, deg) in enumerate(poses):
                writer.writerow(
                    [
                        f"{n:03d}_{i}",
                        format_submission_value(x),
                        format_submission_value(y),
                        format_submission_value(deg),
                    ]
                )

    # Rule of the pipeline: always run overlap validation after generating.
    from santa_packing.scoring import score_submission  # noqa: E402

    _ = score_submission(
        args.out,
        nmax=args.nmax,
        check_overlap=True,
        overlap_mode=str(args.overlap_mode),
        require_complete=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
