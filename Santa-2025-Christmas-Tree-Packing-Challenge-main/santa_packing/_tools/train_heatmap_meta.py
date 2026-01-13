#!/usr/bin/env python3

"""Tool to train a heatmap meta-optimizer with evolution strategies (ES)."""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from santa_packing.geom_np import polygon_radius
from santa_packing.heatmap_meta import HeatmapConfig, heatmap_search, init_params, save_params
from santa_packing.tree_data import TREE_POINTS


def _grid_initial(n: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


def _random_initial(n: int, spacing: float, rand_scale: float, rng: np.random.Generator) -> np.ndarray:
    scale = spacing * math.sqrt(max(n, 1)) * rand_scale
    xy = rng.uniform(-scale, scale, size=(n, 2))
    theta = rng.uniform(0.0, 360.0, size=(n, 1))
    return np.concatenate([xy, theta], axis=1)


def _add_noise(params: Dict[str, np.ndarray], noise: Dict[str, np.ndarray], sigma: float) -> Dict[str, np.ndarray]:
    return {k: params[k] + sigma * noise[k] for k in params}


def _zeros_like(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.zeros_like(v) for k, v in params.items()}


def _accum_grad(accum: Dict[str, np.ndarray], noise: Dict[str, np.ndarray], weight: float) -> None:
    for k in accum:
        accum[k] = accum[k] + noise[k] * weight


def main(argv: list[str] | None = None) -> int:
    """Train heatmap parameters and save them to a `.npz` file."""
    ap = argparse.ArgumentParser(description="Train heatmap meta-optimizer with ES")
    ap.add_argument("--n-list", type=str, default="25,50,100", help="Comma-separated Ns")
    ap.add_argument("--train-steps", type=int, default=50, help="ES iterations")
    ap.add_argument("--es-pop", type=int, default=6, help="ES population size")
    ap.add_argument("--es-sigma", type=float, default=0.05, help="ES noise std")
    ap.add_argument("--lr", type=float, default=0.05, help="ES learning rate")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--policy", type=str, default="gnn", choices=["mlp", "gnn"])
    ap.add_argument("--knn-k", type=int, default=4)
    ap.add_argument("--heatmap-steps", type=int, default=200, help="Search steps per eval")
    ap.add_argument("--t-start", type=float, default=1.0)
    ap.add_argument("--t-end", type=float, default=0.001)
    ap.add_argument("--trans-sigma", type=float, default=0.2)
    ap.add_argument("--rot-sigma", type=float, default=10.0)
    ap.add_argument("--heatmap-lr", type=float, default=0.1)
    ap.add_argument("--init", type=str, default="grid", choices=["grid", "random", "mix"])
    ap.add_argument("--rand-scale", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    if not ns:
        raise SystemExit("Empty --n-list")

    rng = np.random.default_rng(args.seed)
    params = init_params(rng, hidden_size=args.hidden, policy=args.policy)
    config = HeatmapConfig(
        hidden_size=args.hidden,
        policy=args.policy,
        knn_k=args.knn_k,
        heatmap_lr=args.heatmap_lr,
        trans_sigma=args.trans_sigma,
        rot_sigma=args.rot_sigma,
        t_start=args.t_start,
        t_end=args.t_end,
    )

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2

    def eval_params(p, n: int) -> float:
        if args.init == "grid":
            base = _grid_initial(n, spacing)
        elif args.init == "random":
            base = _random_initial(n, spacing, args.rand_scale, rng)
        else:
            base = (
                _grid_initial(n, spacing) if rng.random() < 0.5 else _random_initial(n, spacing, args.rand_scale, rng)
            )
        best_poses, best_score = heatmap_search(p, base, config, args.heatmap_steps, rng)
        return best_score

    for step in range(1, args.train_steps + 1):
        n = random.choice(ns)
        noises = []
        losses = []
        for _ in range(args.es_pop):
            noise = {k: rng.normal(size=v.shape) for k, v in params.items()}
            noisy = _add_noise(params, noise, args.es_sigma)
            loss = eval_params(noisy, n)
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
        out_dir = Path("runs") / f"heatmap_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "heatmap_meta.npz"
    save_params(
        out_path,
        params,
        meta={
            "policy": args.policy,
            "hidden": args.hidden,
            "knn_k": args.knn_k,
            "heatmap_lr": args.heatmap_lr,
            "trans_sigma": args.trans_sigma,
            "rot_sigma": args.rot_sigma,
            "t_start": args.t_start,
            "t_end": args.t_end,
        },
    )
    print(f"Saved heatmap meta model to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
