#!/usr/bin/env python3

"""Scale (expand/shrink) one puzzle `n` about its centroid inside a `submission.csv`.

This is useful as a last-resort "make it safer" tweak when Kaggle flags an overlap
that is hard to reproduce locally due to numeric tolerances. A small scale > 1
increases inter-tree distances and typically resolves borderline overlaps/touches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from santa_packing.cli.generate_submission import _finalize_puzzle
from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score
from santa_packing.scoring import OverlapMode, load_submission, score_submission
from santa_packing.submission_format import fit_xy_in_bounds, quantize_for_submission
from santa_packing.tree_data import TREE_POINTS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scale puzzle n about centroid inside a submission.csv.")
    ap.add_argument("--base", type=Path, required=True, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--n", type=int, required=True, help="Puzzle id (1..200)")
    ap.add_argument("--scale", type=float, required=True, help="Scale factor (e.g. 1.005)")
    ap.add_argument("--seed", type=int, default=123, help="Seed for finalize/repair randomness")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="kaggle",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate enforced during finalize (default: kaggle).",
    )
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="Run finalize/repair after scaling (recommended).",
    )
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n in the CSV (default: 200)")
    ap.add_argument("--score-total", action="store_true", help="Compute total score (slower).")
    ns = ap.parse_args(argv)

    n = int(ns.n)
    if n < 1 or n > 200:
        raise SystemExit("--n must be in [1,200]")

    nmax = int(ns.nmax)
    if nmax < n or nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [n,200]")

    base = Path(ns.base).resolve()
    if not base.is_file():
        raise SystemExit(f"base submission not found: {base}")

    puzzles = load_submission(base, nmax=nmax)
    missing = [k for k in range(1, nmax + 1) if k not in puzzles or puzzles[k].shape != (k, 3)]
    if missing:
        raise SystemExit(f"submission missing/invalid puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    scale = float(ns.scale)
    if not np.isfinite(scale) or scale <= 0:
        raise SystemExit("--scale must be a positive finite float")

    points = np.array(TREE_POINTS, dtype=float)
    overlap_mode: OverlapMode = str(ns.overlap_mode)  # type: ignore[assignment]

    poses = np.array(puzzles[n], dtype=float, copy=True)
    s_before = float(packing_score(points, poses))

    center = np.mean(poses[:, 0:2], axis=0)
    poses[:, 0:2] = center[None, :] + (poses[:, 0:2] - center[None, :]) * scale
    poses[:, 2] = np.mod(poses[:, 2], 360.0)

    poses_q = quantize_for_submission(fit_xy_in_bounds(poses))
    if bool(ns.finalize):
        poses_q = _finalize_puzzle(
            points,
            poses_q,
            seed=int(ns.seed) + 1_000_003 * int(n),
            puzzle_n=int(n),
            overlap_mode=overlap_mode,
        )

    s_after = float(packing_score(points, poses_q))
    print(f"n={n} scale={scale:.12g} s_before={s_before:.12f} s_after={s_after:.12f} delta={s_before - s_after:+.12f}")

    puzzles[n] = np.array(poses_q, dtype=float, copy=True)
    out = Path(ns.out).resolve()
    _write_submission(out, puzzles, nmax=nmax)
    print(f"wrote: {out}")

    if bool(ns.score_total):
        res = score_submission(out, nmax=nmax, check_overlap=True, overlap_mode=str(overlap_mode), require_complete=True)
        print(f"total_score={res.score:.12f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

