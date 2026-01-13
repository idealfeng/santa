#!/usr/bin/env python3

"""Tool to train an L2O policy via behavior cloning from an SA dataset."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from santa_packing.l2o import (
    L2OConfig,
    behavior_cloning_loss,
    behavior_cloning_loss_weighted,
    init_params,
    save_params_npz,
)


def _adam_update(params, grads, opt_state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    step, m, v = opt_state
    step += 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), v, grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1**step), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2**step), v)
    params = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), params, m_hat, v_hat)
    return params, (step, m, v)


def main(argv: list[str] | None = None) -> int:
    """Train a policy from a dataset of accepted SA moves and save it to `--out`."""
    ap = argparse.ArgumentParser(description="Train L2O by behavior cloning from SA dataset")
    ap.add_argument("--dataset", type=Path, required=True, help="Dataset .npz from collect_sa_dataset.py")
    ap.add_argument("--n-list", type=str, default="", help="Comma-separated Ns (default: infer)")
    ap.add_argument("--batch", type=int, default=64, help="Batch size")
    ap.add_argument("--train-steps", type=int, default=500, help="Training iterations")
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
    ap.add_argument("--trans-sigma", type=float, default=0.2, help="BC translation sigma")
    ap.add_argument("--rot-sigma", type=float, default=10.0, help="BC rotation sigma")
    ap.add_argument("--curriculum", action="store_true", help="Enable curriculum over N (start small, grow)")
    ap.add_argument("--curriculum-start-max", type=int, default=None, help="Max N at start (default: min(n_list))")
    ap.add_argument("--curriculum-end-max", type=int, default=None, help="Max N at end (default: max(n_list))")
    ap.add_argument(
        "--curriculum-steps", type=int, default=None, help="Steps to ramp curriculum (default: train_steps)"
    )
    ap.add_argument("--out", type=Path, default=None, help="Output policy path (.npz)")
    args = ap.parse_args(argv)
    if args.reward is None:
        args.reward = "prefix" if args.preset == "submission" else "packing"

    data = np.load(args.dataset)
    if args.n_list.strip():
        ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    else:
        ns = sorted({int(k.split("n")[1]) for k in data.files if k.startswith("poses_n")})
    if not ns:
        raise SystemExit("No N values found in dataset.")

    pools = {}
    for n in ns:
        poses = data[f"poses_n{n}"]
        idxs = data[f"idx_n{n}"]
        deltas = data[f"delta_n{n}"]
        dscores = data[f"dscore_n{n}"] if f"dscore_n{n}" in data.files else None
        pools[n] = (poses, idxs, deltas, dscores)

    key = jax.random.PRNGKey(args.seed)
    params = init_params(
        key,
        hidden_size=args.hidden,
        policy=args.policy,
        mlp_depth=args.mlp_depth,
        gnn_steps=args.gnn_steps,
        gnn_attention=args.gnn_attention,
        feature_mode=args.feature_mode,
    )
    config = L2OConfig(
        hidden_size=args.hidden,
        policy=args.policy,
        knn_k=args.knn_k,
        feature_mode=args.feature_mode,
        reward=args.reward,
        mlp_depth=args.mlp_depth,
        gnn_steps=args.gnn_steps,
        gnn_attention=args.gnn_attention,
        trans_sigma=args.trans_sigma,
        rot_sigma=args.rot_sigma,
    )

    opt_state = (
        0,
        jax.tree_util.tree_map(jnp.zeros_like, params),
        jax.tree_util.tree_map(jnp.zeros_like, params),
    )

    use_weights = any(pools[n][3] is not None for n in ns)
    if use_weights:
        loss_grad = jax.value_and_grad(
            lambda p, poses, idxs, deltas, w: behavior_cloning_loss_weighted(p, poses, idxs, deltas, w, config)
        )
    else:
        loss_grad = jax.value_and_grad(
            lambda p, poses, idxs, deltas: behavior_cloning_loss(p, poses, idxs, deltas, config)
        )

    n_min = min(ns) if ns else 1
    n_max = max(ns) if ns else n_min
    curriculum_start_max = n_min if args.curriculum_start_max is None else int(args.curriculum_start_max)
    curriculum_end_max = n_max if args.curriculum_end_max is None else int(args.curriculum_end_max)
    curriculum_steps = args.train_steps if args.curriculum_steps is None else int(args.curriculum_steps)

    for step in range(1, args.train_steps + 1):
        if args.curriculum:
            frac = min(step / max(curriculum_steps, 1), 1.0)
            allowed_max = int(round(curriculum_start_max + frac * (curriculum_end_max - curriculum_start_max)))
            allowed = [n for n in ns if n <= allowed_max]
            n = random.choice(allowed) if allowed else n_min
        else:
            n = random.choice(ns)
        poses_pool, idx_pool, delta_pool, dscore_pool = pools[n]
        idx = np.random.randint(0, poses_pool.shape[0], size=args.batch)
        poses_batch = jnp.array(poses_pool[idx])
        idx_batch = jnp.array(idx_pool[idx])
        delta_batch = jnp.array(delta_pool[idx])
        if use_weights:
            w_batch = jnp.ones((args.batch,), dtype=jnp.float32)
            if dscore_pool is not None:
                w_batch = jnp.array(dscore_pool[idx])
            loss, grads = loss_grad(params, poses_batch, idx_batch, delta_batch, w_batch)
        else:
            loss, grads = loss_grad(params, poses_batch, idx_batch, delta_batch)
        params, opt_state = _adam_update(params, grads, opt_state, lr=args.lr)
        if step % 20 == 0 or step == 1:
            print(f"[{step:04d}] loss={float(loss):.6f}")

    out_path = args.out
    if out_path is None:
        out_dir = Path("runs") / f"l2o_bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            "trans_sigma": args.trans_sigma,
            "rot_sigma": args.rot_sigma,
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
