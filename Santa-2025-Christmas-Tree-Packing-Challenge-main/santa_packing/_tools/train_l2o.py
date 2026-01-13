#!/usr/bin/env python3

"""Tool to train an L2O policy (REINFORCE) for packing refinement."""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from santa_packing.geom_np import polygon_radius, shift_poses_to_origin
from santa_packing.l2o import L2OConfig, init_params, loss_fn, loss_with_baseline, save_params_npz
from santa_packing.lattice import lattice_poses
from santa_packing.tree_data import TREE_POINTS


def _grid_initial(n: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


def _random_initial(n: int, spacing: float, rand_scale: float) -> np.ndarray:
    scale = spacing * math.sqrt(max(n, 1)) * rand_scale
    xy = np.random.uniform(-scale, scale, size=(n, 2))
    theta = np.random.uniform(0.0, 360.0, size=(n, 1))
    return np.concatenate([xy, theta], axis=1)


def _lattice_initial(n: int, *, pattern: str, margin: float, rotate_deg: float) -> np.ndarray:
    return lattice_poses(n, pattern=pattern, margin=margin, rotate_deg=rotate_deg)


def _make_dataset(
    ns: list[int],
    per_n: int,
    spacing: float,
    init_mode: str,
    rand_scale: float,
    *,
    lattice_pattern: str = "hex",
    lattice_margin: float = 0.02,
    lattice_rotate: float = 0.0,
) -> dict[int, np.ndarray]:
    dataset: dict[int, np.ndarray] = {}
    for n in ns:
        samples = np.zeros((per_n, n, 3), dtype=float)
        for i in range(per_n):
            if init_mode == "grid":
                poses = shift_poses_to_origin(np.array(TREE_POINTS, dtype=float), _grid_initial(n, spacing))
            elif init_mode == "random":
                poses = shift_poses_to_origin(
                    np.array(TREE_POINTS, dtype=float), _random_initial(n, spacing, rand_scale)
                )
            elif init_mode == "lattice":
                poses = _lattice_initial(n, pattern=lattice_pattern, margin=lattice_margin, rotate_deg=lattice_rotate)
            else:
                # mix/all: cycle across grid/random/(optional) lattice for diversity
                if init_mode == "mix":
                    mode = i % 2
                    if mode == 0:
                        poses = shift_poses_to_origin(np.array(TREE_POINTS, dtype=float), _grid_initial(n, spacing))
                    else:
                        poses = shift_poses_to_origin(
                            np.array(TREE_POINTS, dtype=float),
                            _random_initial(n, spacing, rand_scale),
                        )
                else:
                    mode = i % 3
                    if mode == 0:
                        poses = shift_poses_to_origin(np.array(TREE_POINTS, dtype=float), _grid_initial(n, spacing))
                    elif mode == 1:
                        poses = shift_poses_to_origin(
                            np.array(TREE_POINTS, dtype=float),
                            _random_initial(n, spacing, rand_scale),
                        )
                    else:
                        poses = _lattice_initial(
                            n, pattern=lattice_pattern, margin=lattice_margin, rotate_deg=lattice_rotate
                        )
            samples[i] = np.array(poses, dtype=float)
        dataset[n] = samples
    return dataset


def _adam_update(params, grads, opt_state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    step, m, v = opt_state
    step += 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), v, grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1**step), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2**step), v)
    params = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), params, m_hat, v_hat)
    return params, (step, m, v)


def train_l2o_model(
    *,
    seed: int = 1,
    n_list: list[int],
    batch: int = 64,
    train_steps: int = 200,
    steps: int = 200,
    lr: float = 1e-3,
    hidden_size: int = 32,
    mlp_depth: int = 1,
    gnn_steps: int = 1,
    gnn_attention: bool = False,
    policy: str = "mlp",
    knn_k: int = 4,
    feature_mode: str = "raw",
    reward: str = "packing",
    action_scale: float = 1.0,
    overlap_penalty: float = 50.0,
    overlap_lambda: float = 0.0,
    init_mode: str = "grid",
    rand_scale: float = 0.3,
    lattice_pattern: str = "hex",
    lattice_margin: float = 0.02,
    lattice_rotate: float = 0.0,
    dataset: dict[int, np.ndarray] | None = None,
    verbose_freq: int = 20,
    baseline_mode: str = "batch",
    baseline_decay: float = 0.9,
    curriculum: bool = False,
    curriculum_start_max: int | None = None,
    curriculum_end_max: int | None = None,
    curriculum_steps: int | None = None,
):
    """Train an L2O policy (REINFORCE) for packing refinement.

    Most parameters map 1:1 to CLI flags in `main()`. The function samples
    initial states (grid/random/lattice or from a provided dataset) and optimizes
    a policy network to minimize the chosen reward objective.

    Returns:
        Tuple `(params, history)` where `params` is the trained model parameters
        (a JAX pytree) and `history` is a list of loss values per training step.
    """
    key = jax.random.PRNGKey(seed)
    params = init_params(
        key,
        hidden_size=hidden_size,
        policy=policy,
        mlp_depth=mlp_depth,
        gnn_steps=gnn_steps,
        gnn_attention=gnn_attention,
        feature_mode=feature_mode,
    )

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2

    # Init Optimizer
    opt_state = (
        0,
        jax.tree_util.tree_map(jnp.zeros_like, params),
        jax.tree_util.tree_map(jnp.zeros_like, params),
    )

    config = L2OConfig(
        hidden_size=hidden_size,
        policy=policy,
        knn_k=knn_k,
        feature_mode=feature_mode,
        reward=reward,
        mlp_depth=mlp_depth,
        gnn_steps=gnn_steps,
        gnn_attention=gnn_attention,
        action_scale=action_scale,
        overlap_penalty=overlap_penalty,
        overlap_lambda=overlap_lambda,
    )
    baseline = None
    if baseline_mode == "ema":
        loss_grad = jax.value_and_grad(
            lambda p, k, batch, b: loss_with_baseline(p, k, batch, steps, config, b),
            has_aux=True,
        )
    else:
        loss_grad = jax.value_and_grad(lambda p, k, batch: loss_fn(p, k, batch, steps, config))
    loss_grad = jax.jit(loss_grad)

    history = []

    n_min = min(n_list) if n_list else 1
    n_max = max(n_list) if n_list else n_min
    if curriculum_start_max is None:
        curriculum_start_max = n_min
    if curriculum_end_max is None:
        curriculum_end_max = n_max
    if curriculum_steps is None:
        curriculum_steps = train_steps

    for step in range(1, train_steps + 1):
        if curriculum:
            frac = min(step / max(int(curriculum_steps), 1), 1.0)
            allowed_max = int(round(curriculum_start_max + frac * (curriculum_end_max - curriculum_start_max)))
            allowed = [n for n in n_list if n <= allowed_max]
            n = random.choice(allowed) if allowed else n_min
        else:
            n = random.choice(n_list)
        if dataset is None:
            samples = np.zeros((batch, n, 3), dtype=float)
            for i in range(batch):
                if init_mode == "grid":
                    poses = shift_poses_to_origin(points, _grid_initial(n, spacing))
                elif init_mode == "random":
                    poses = shift_poses_to_origin(points, _random_initial(n, spacing, rand_scale))
                elif init_mode == "lattice":
                    poses = _lattice_initial(
                        n, pattern=lattice_pattern, margin=lattice_margin, rotate_deg=lattice_rotate
                    )
                else:
                    if init_mode == "mix":
                        mode = i % 2
                        if mode == 0:
                            poses = shift_poses_to_origin(points, _grid_initial(n, spacing))
                        else:
                            poses = shift_poses_to_origin(points, _random_initial(n, spacing, rand_scale))
                    else:
                        mode = i % 3
                        if mode == 0:
                            poses = shift_poses_to_origin(points, _grid_initial(n, spacing))
                        elif mode == 1:
                            poses = shift_poses_to_origin(points, _random_initial(n, spacing, rand_scale))
                        else:
                            poses = _lattice_initial(
                                n, pattern=lattice_pattern, margin=lattice_margin, rotate_deg=lattice_rotate
                            )
                samples[i] = np.array(poses, dtype=float)
        else:
            pool = dataset[n]
            idx = np.random.randint(0, pool.shape[0], size=batch)
            samples = pool[idx]

        poses_batch = jnp.array(samples)
        key, sub = jax.random.split(key)
        if baseline_mode == "ema":
            baseline_val = 0.0 if baseline is None else baseline
            (loss, reward_mean), grads = loss_grad(params, sub, poses_batch, baseline_val)
            reward_mean = float(reward_mean)
            if baseline is None:
                baseline = reward_mean
            else:
                baseline = baseline_decay * baseline + (1.0 - baseline_decay) * reward_mean
        else:
            loss, grads = loss_grad(params, sub, poses_batch)
        params, opt_state = _adam_update(params, grads, opt_state, lr=lr)

        loss_val = float(loss)
        history.append(loss_val)

        if verbose_freq > 0 and (step % verbose_freq == 0 or step == 1):
            print(f"[{step:04d}] loss={loss_val:.6f}")

    return params, history


def main(argv: list[str] | None = None) -> int:
    """Train an L2O policy and save it to `--out`."""
    ap = argparse.ArgumentParser(description="Train a minimal L2O policy (REINFORCE)")
    ap.add_argument("--n", type=int, default=10, help="Number of trees (fallback if --n-list is empty)")
    ap.add_argument("--n-list", type=str, default="", help="Comma-separated list of N values (ex: 25,50,100)")
    ap.add_argument("--steps", type=int, default=200, help="Rollout steps")
    ap.add_argument("--batch", type=int, default=64, help="Batch size")
    ap.add_argument("--train-steps", type=int, default=200, help="Training iterations")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--seed", type=int, default=1, help="Random seed")
    ap.add_argument("--hidden", type=int, default=32, help="Hidden size")
    ap.add_argument("--mlp-depth", type=int, default=1, help="MLP depth (>=1)")
    ap.add_argument("--gnn-steps", type=int, default=1, help="GNN message passing steps")
    ap.add_argument("--gnn-attention", action="store_true", help="Enable attention aggregation in GNN")
    ap.add_argument("--policy", type=str, default="mlp", choices=["mlp", "gnn"], help="Policy backbone")
    ap.add_argument("--knn-k", type=int, default=4, help="KNN neighbors for GNN")
    ap.add_argument(
        "--feature-mode",
        type=str,
        default="raw",
        choices=["raw", "bbox_norm", "rich"],
        help="Input feature representation",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "submission"],
        help="Preset for defaults. 'submission' aligns reward/objective with the official prefix score.",
    )
    ap.add_argument(
        "--reward",
        type=str,
        default=None,
        choices=["packing", "prefix"],
        help="Reward type. Default: packing (preset=default) or prefix (preset=submission).",
    )
    ap.add_argument("--action-scale", type=float, default=1.0, help="Scale applied to policy mean actions")
    ap.add_argument(
        "--overlap-penalty",
        type=float,
        default=50.0,
        help="Penalty for any collision (binary). Set 0 to disable collision check in reward.",
    )
    ap.add_argument(
        "--overlap-lambda",
        type=float,
        default=0.0,
        help="Weight for circle-overlap penalty (smooth-ish proxy). 0 disables.",
    )
    ap.add_argument("--baseline", type=str, default="batch", choices=["batch", "ema"], help="Baseline mode")
    ap.add_argument("--baseline-decay", type=float, default=0.9, help="EMA decay for baseline")
    ap.add_argument(
        "--init", type=str, default="grid", choices=["grid", "random", "mix", "lattice", "all"], help="Init poses"
    )
    ap.add_argument("--rand-scale", type=float, default=0.3, help="Random init scale (relative)")
    ap.add_argument("--lattice-pattern", type=str, default="hex", choices=["hex", "square"], help="Lattice pattern")
    ap.add_argument("--lattice-margin", type=float, default=0.02, help="Lattice margin")
    ap.add_argument("--lattice-rotate", type=float, default=0.0, help="Lattice rotation (deg)")
    ap.add_argument("--curriculum", action="store_true", help="Enable curriculum over N (start small, grow)")
    ap.add_argument("--curriculum-start-max", type=int, default=None, help="Max N at start (default: min(n_list))")
    ap.add_argument("--curriculum-end-max", type=int, default=None, help="Max N at end (default: max(n_list))")
    ap.add_argument(
        "--curriculum-steps", type=int, default=None, help="Steps to ramp curriculum (default: train_steps)"
    )
    ap.add_argument("--dataset-size", type=int, default=0, help="Pre-generate dataset per N (0 = on-the-fly)")
    ap.add_argument("--dataset-out", type=Path, default=None, help="Optional dataset output (.npz)")
    ap.add_argument("--dataset-in", type=Path, default=None, help="Optional dataset input (.npz)")
    ap.add_argument("--out", type=Path, default=None, help="Output policy path (.npz)")
    args = ap.parse_args(argv)
    if args.reward is None:
        args.reward = "prefix" if args.preset == "submission" else "packing"

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2

    if args.n_list.strip():
        ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    else:
        ns = [args.n]

    dataset: dict[int, np.ndarray] | None = None
    if args.dataset_in and args.dataset_in.exists():
        data = np.load(args.dataset_in)
        dataset = {}
        for n in ns:
            key_name = f"poses_n{n}"
            if key_name in data:
                dataset[n] = data[key_name]
        if not dataset:
            dataset = None

    if dataset is None and args.dataset_size > 0:
        dataset = _make_dataset(
            ns,
            args.dataset_size,
            spacing,
            args.init,
            args.rand_scale,
            lattice_pattern=args.lattice_pattern,
            lattice_margin=args.lattice_margin,
            lattice_rotate=args.lattice_rotate,
        )
        if args.dataset_out:
            payload = {f"poses_n{n}": dataset[n] for n in dataset}
            np.savez(args.dataset_out, **payload)
            print(f"Saved dataset to {args.dataset_out}")

    params, _ = train_l2o_model(
        seed=args.seed,
        n_list=ns,
        batch=args.batch,
        train_steps=args.train_steps,
        steps=args.steps,
        lr=args.lr,
        hidden_size=args.hidden,
        mlp_depth=args.mlp_depth,
        gnn_steps=args.gnn_steps,
        gnn_attention=args.gnn_attention,
        policy=args.policy,
        knn_k=args.knn_k,
        feature_mode=args.feature_mode,
        reward=args.reward,
        action_scale=args.action_scale,
        overlap_penalty=args.overlap_penalty,
        overlap_lambda=args.overlap_lambda,
        init_mode=args.init,
        rand_scale=args.rand_scale,
        lattice_pattern=args.lattice_pattern,
        lattice_margin=args.lattice_margin,
        lattice_rotate=args.lattice_rotate,
        dataset=dataset,
        baseline_mode=args.baseline,
        baseline_decay=args.baseline_decay,
        curriculum=args.curriculum,
        curriculum_start_max=args.curriculum_start_max,
        curriculum_end_max=args.curriculum_end_max,
        curriculum_steps=args.curriculum_steps,
    )

    out_path = args.out
    if out_path is None:
        out_dir = Path("runs") / f"l2o_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "policy.npz"
    save_params_npz(
        out_path,
        params,
        meta={
            "policy": args.policy,
            "hidden": args.hidden,
            "knn_k": args.knn_k,
            "mlp_depth": args.mlp_depth,
            "gnn_steps": args.gnn_steps,
            "gnn_attention": args.gnn_attention,
            "feature_mode": args.feature_mode,
            "reward": args.reward,
            "action_scale": args.action_scale,
            "overlap_penalty": args.overlap_penalty,
            "overlap_lambda": args.overlap_lambda,
            "init_mode": args.init,
            "rand_scale": args.rand_scale,
            "lattice_pattern": args.lattice_pattern,
            "lattice_margin": args.lattice_margin,
            "lattice_rotate": args.lattice_rotate,
            "curriculum": bool(args.curriculum),
            "curriculum_start_max": args.curriculum_start_max,
            "curriculum_end_max": args.curriculum_end_max,
            "curriculum_steps": args.curriculum_steps,
        },
    )
    print(f"Saved policy to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
