#!/usr/bin/env python3

"""Tool to train a meta-initialization policy with ES + fixed SA steps."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from santa_packing.collisions import check_any_collisions
from santa_packing.geom_np import polygon_radius, shift_poses_to_origin
from santa_packing.meta_init import MetaInitConfig, apply_meta_init, init_meta_params, save_meta_params
from santa_packing.optimizer import run_sa_batch
from santa_packing.tree import get_tree_polygon
from santa_packing.tree_data import TREE_POINTS


def _grid_initial(n: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


def _add_noise(params: Dict[str, jnp.ndarray], noise: Dict[str, jnp.ndarray], sigma: float) -> Dict[str, jnp.ndarray]:
    return {k: params[k] + sigma * noise[k] for k in params}


def _zeros_like(params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.zeros_like(v) for k, v in params.items()}


def _accum_grad(accum: Dict[str, jnp.ndarray], noise: Dict[str, jnp.ndarray], weight: float) -> None:
    for k in accum:
        accum[k] = accum[k] + noise[k] * weight


def main(argv: list[str] | None = None) -> int:
    """Train meta-init parameters and save them to a `.npz` file."""
    ap = argparse.ArgumentParser(description="Meta-train initialization policy with ES + fixed SA steps")
    ap.add_argument("--n-list", type=str, default="25,50,100", help="Comma-separated Ns")
    ap.add_argument(
        "--objective", type=str, default="packing", choices=["packing", "prefix"], help="SA objective used for training"
    )
    ap.add_argument("--train-steps", type=int, default=50, help="Outer ES iterations")
    ap.add_argument("--es-pop", type=int, default=6, help="ES population size")
    ap.add_argument("--es-sigma", type=float, default=0.05, help="ES noise std")
    ap.add_argument("--lr", type=float, default=0.05, help="ES learning rate")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--delta-xy", type=float, default=0.2)
    ap.add_argument("--delta-theta", type=float, default=10.0)
    ap.add_argument("--sa-steps", type=int, default=100, help="SA steps for inner loop")
    ap.add_argument("--sa-batch", type=int, default=16, help="SA batch size")
    ap.add_argument("--sa-trans-sigma", type=float, default=0.1)
    ap.add_argument("--sa-rot-sigma", type=float, default=15.0)
    ap.add_argument("--sa-rot-prob", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None, help="Output meta-init model (.npz)")
    args = ap.parse_args(argv)

    ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    if not ns:
        raise SystemExit("Empty --n-list")

    key = jax.random.PRNGKey(args.seed)
    params = init_meta_params(key, hidden_size=args.hidden)
    config = MetaInitConfig(hidden_size=args.hidden, delta_xy=args.delta_xy, delta_theta=args.delta_theta)

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2
    base_poly = get_tree_polygon()

    def eval_params(p, key, n: int) -> float:
        base = _grid_initial(n, spacing)
        base = shift_poses_to_origin(points, base)
        init = apply_meta_init(p, jnp.array(base), config)
        # IMPORTANT: run_sa_batch assumes the starting state is collision-free.
        # If init is colliding, the incremental one-vs-all collision tracking can
        # miss pre-existing overlaps. Penalize and skip SA in that case.
        if bool(check_any_collisions(init, base_poly)):
            return 1000.0
        init_batch = jnp.tile(init[None, :, :], (args.sa_batch, 1, 1))
        best_poses, best_scores = run_sa_batch(
            key,
            args.sa_steps,
            n,
            init_batch,
            trans_sigma=args.sa_trans_sigma,
            rot_sigma=args.sa_rot_sigma,
            rot_prob=args.sa_rot_prob,
            objective=args.objective,
        )
        best_scores.block_until_ready()
        return float(jnp.min(best_scores))

    for step in range(1, args.train_steps + 1):
        n = random.choice(ns)
        key, sub = jax.random.split(key)
        noises = []
        losses = []
        for _ in range(args.es_pop):
            sub, nkey = jax.random.split(sub)
            noise = {k: jax.random.normal(nkey, v.shape) for k, v in params.items()}
            noisy = _add_noise(params, noise, args.es_sigma)
            loss = eval_params(noisy, sub, n)
            noises.append(noise)
            losses.append(loss)

        losses_np = np.array(losses, dtype=float)
        if np.std(losses_np) > 1e-9:
            norm_losses = (losses_np - losses_np.mean()) / (losses_np.std() + 1e-9)
        else:
            norm_losses = losses_np - losses_np.mean()

        grad = _zeros_like(params)
        for noise, weight in zip(noises, norm_losses):
            _accum_grad(grad, noise, weight)
        for k in params:
            params[k] = params[k] - (args.lr / (args.es_pop * args.es_sigma)) * grad[k]

        if step % 5 == 0 or step == 1:
            print(f"[{step:03d}] n={n} loss_mean={float(np.mean(losses_np)):.6f}")

    out_path = args.out
    if out_path is None:
        out_dir = Path("runs") / f"meta_init_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "meta_init.npz"
    save_meta_params(
        out_path,
        params,
        meta={
            "hidden": args.hidden,
            "delta_xy": args.delta_xy,
            "delta_theta": args.delta_theta,
            "objective": args.objective,
            "sa_steps": args.sa_steps,
            "sa_batch": args.sa_batch,
            "sa_trans_sigma": args.sa_trans_sigma,
            "sa_rot_sigma": args.sa_rot_sigma,
            "sa_rot_prob": args.sa_rot_prob,
        },
    )
    print(f"Saved meta-init model to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
