#!/usr/bin/env python3

"""Multi-seed JAX refinement + per-puzzle ensembling for an existing submission.

Why this exists:
- The repo already ships strong `submission.csv` solutions.
- Running `generate_submission` separately per seed is slow on Windows because JAX will
  recompile kernels in every process.
- This tool keeps everything in ONE Python process so JAX compilation is reused across
  seeds, then selects the best (lowest `s_n`) packing per puzzle `n`.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.cli.generate_submission import _finalize_puzzle, _run_sa
from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score
from santa_packing.scoring import first_overlap_pair, load_submission, score_submission
from santa_packing.tree_data import TREE_POINTS


def _parse_int_list(text: str) -> list[int]:
    raw = text.strip()
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..", 1)
            start = int(a)
            end = int(b)
            step = 1 if end >= start else -1
            out.extend(list(range(start, end + step, step)))
            continue
        if "-" in part and part.count("-") == 1 and part[0] != "-":
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
            continue
        out.append(int(part))
    return out


@dataclass(frozen=True)
class RefineConfig:
    steps: int
    batch: int
    trans_sigma: float
    rot_sigma: float
    rot_prob: float
    rot_prob_end: float
    proposal: str
    neighborhood: bool
    objective: str
    overlap_lambda: float
    allow_collisions: bool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refine an existing submission with JAX SA across many seeds, ensemble per n.")
    ap.add_argument("--base", type=Path, required=True, help="Baseline submission.csv to refine (must contain 1..nmax).")
    ap.add_argument("--out", type=Path, required=True, help="Output ensembled submission.csv")
    ap.add_argument("--nmax", type=int, default=200)
    ap.add_argument("--overlap-mode", type=str, default="kaggle", choices=["strict", "conservative", "kaggle"])
    ap.add_argument("--seeds", type=str, default="1", help="Comma list or ranges, e.g. 1,2,3 or 1000..1015")
    ap.add_argument("--n-min", type=int, default=1, help="Only attempt refinement for n >= this (default: 1)")
    ap.add_argument("--n-max", type=int, default=None, help="Only attempt refinement for n <= this (default: nmax)")
    ap.add_argument("--tol", type=float, default=1e-12, help="Strict improvement tolerance for s_n")

    ap.add_argument("--steps", type=int, default=4000, help="SA steps per puzzle")
    ap.add_argument("--batch", type=int, default=32, help="SA batch size per puzzle")
    ap.add_argument("--trans-sigma", type=float, default=0.08)
    ap.add_argument("--rot-sigma", type=float, default=8.0)
    ap.add_argument("--rot-prob", type=float, default=0.15)
    ap.add_argument("--rot-prob-end", type=float, default=0.05)
    ap.add_argument("--proposal", type=str, default="mixed", choices=["random", "bbox_inward", "bbox", "inward", "smart", "mixed"])
    ap.add_argument("--neighborhood", action="store_true", help="Enable swap/compact/teleport moves with defaults")
    ap.add_argument("--objective", type=str, default="packing", choices=["packing", "prefix"])
    ap.add_argument("--overlap-lambda", type=float, default=0.0, help="Circle-overlap penalty (use with --allow-collisions)")
    ap.add_argument("--allow-collisions", action="store_true", help="Allow accepting colliding states during SA (finalize will repair)")

    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    seeds = _parse_int_list(ns.seeds)
    if not seeds:
        raise SystemExit("Empty --seeds")

    n_min = int(ns.n_min)
    n_max = int(ns.n_max) if ns.n_max is not None else nmax
    if n_min < 1 or n_min > nmax:
        raise SystemExit("--n-min must be in [1,nmax]")
    if n_max < n_min or n_max > nmax:
        raise SystemExit("--n-max must be in [n-min,nmax]")

    cfg = RefineConfig(
        steps=int(ns.steps),
        batch=int(ns.batch),
        trans_sigma=float(ns.trans_sigma),
        rot_sigma=float(ns.rot_sigma),
        rot_prob=float(ns.rot_prob),
        rot_prob_end=float(ns.rot_prob_end),
        proposal=str(ns.proposal),
        neighborhood=bool(ns.neighborhood),
        objective=str(ns.objective),
        overlap_lambda=float(ns.overlap_lambda),
        allow_collisions=bool(ns.allow_collisions),
    )

    base_csv = ns.base.resolve()
    if not base_csv.is_file():
        raise SystemExit(f"Base submission not found: {base_csv}")

    puzzles = load_submission(base_csv, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in puzzles or puzzles[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"Base CSV missing/invalid puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    points = np.array(TREE_POINTS, dtype=float)

    best: dict[int, np.ndarray] = {n: np.array(puzzles[n], dtype=float, copy=True) for n in range(1, nmax + 1)}
    best_s: dict[int, float] = {n: float(packing_score(points, best[n])) for n in range(1, nmax + 1)}

    t0 = time.time()
    improved_total = 0

    for run_idx, seed in enumerate(seeds, start=1):
        improved_this_seed = 0
        for n in range(n_min, n_max + 1):
            base_poses = best[n]
            sa_seed = int(seed) + 1_000_003 * int(n)

            refined = _run_sa(
                int(n),
                seed=int(sa_seed),
                batch_size=int(cfg.batch),
                n_steps=int(cfg.steps),
                trans_sigma=float(cfg.trans_sigma),
                rot_sigma=float(cfg.rot_sigma),
                rot_prob=float(cfg.rot_prob),
                rot_prob_end=float(cfg.rot_prob_end),
                swap_prob=0.0,
                swap_prob_end=-1.0,
                push_prob=0.15 if cfg.neighborhood else 0.0,
                push_scale=1.0,
                push_square_prob=0.5,
                compact_prob=0.12 if cfg.neighborhood else 0.0,
                compact_prob_end=0.25 if cfg.neighborhood else -1.0,
                compact_scale=1.0,
                compact_square_prob=0.75,
                teleport_prob=0.05 if cfg.neighborhood else 0.0,
                teleport_prob_end=0.01 if cfg.neighborhood else -1.0,
                teleport_tries=4,
                teleport_anchor_beta=6.0,
                teleport_ring_mult=1.02,
                teleport_jitter=0.05,
                cooling="geom",
                cooling_power=1.0,
                trans_sigma_nexp=0.0,
                rot_sigma_nexp=0.0,
                sigma_nref=float(max(50.0, nmax)),
                proposal=str(cfg.proposal),
                smart_prob=1.0,
                smart_beta=8.0,
                smart_drift=1.0,
                smart_noise=0.25,
                overlap_lambda=float(cfg.overlap_lambda),
                allow_collisions=bool(cfg.allow_collisions),
                initial_poses=base_poses,
                objective=str(cfg.objective),
            )
            if refined is None:
                continue

            finalized = _finalize_puzzle(
                points,
                refined,
                seed=int(sa_seed) + 13_000,
                puzzle_n=int(n),
                overlap_mode=str(ns.overlap_mode),
            )
            if first_overlap_pair(points, finalized, mode=str(ns.overlap_mode)) is not None:
                continue

            s = float(packing_score(points, finalized))
            if s + float(ns.tol) < float(best_s[n]):
                best[n] = np.array(finalized, dtype=float, copy=True)
                best_s[n] = float(s)
                improved_total += 1
                improved_this_seed += 1

        dt = time.time() - t0
        print(
            f"[{run_idx:03d}/{len(seeds):03d}] seed={seed} improved={improved_this_seed} total_improved={improved_total} elapsed_s={dt:.1f}",
            flush=True,
        )

    out_path = ns.out.resolve()
    _write_submission(out_path, best, nmax=nmax)

    res = score_submission(out_path, nmax=nmax, check_overlap=True, overlap_mode=str(ns.overlap_mode), require_complete=True)
    print(f"wrote: {out_path}")
    print(f"score: {res.score:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

