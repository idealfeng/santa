#!/usr/bin/env python3

"""Apply manual tweaks to one puzzle `n` inside a `submission.csv`.

Workflow:
1) Visualize: `python -m santa_packing._tools.viz_puzzle --n 20 --label --highlight-boundary`
2) Pick one or two boundary trees.
3) Apply small moves here and immediately validate/score.

Notes:
- This keeps the rest of the submission unchanged.
- The tweaked puzzle is finalized (quantized + bounds fit + repair if needed) to
  avoid producing an invalid CSV by accident.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from santa_packing.cli.generate_submission import _finalize_puzzle
from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score
from santa_packing.scoring import OverlapMode, first_overlap_pair, load_submission, score_submission
from santa_packing.submission_format import fit_xy_in_bounds, quantize_for_submission
from santa_packing.tree_data import TREE_POINTS


def _parse_move(text: str) -> tuple[int, float, float, float]:
    # idx,dx,dy[,ddeg]
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) not in {3, 4}:
        raise ValueError("move must be idx,dx,dy[,ddeg]")
    idx = int(parts[0])
    dx = float(parts[1])
    dy = float(parts[2])
    ddeg = float(parts[3]) if len(parts) == 4 else 0.0
    return idx, dx, dy, ddeg


def _parse_set(text: str) -> tuple[int, float, float, float]:
    # idx,x,y[,deg]
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) not in {3, 4}:
        raise ValueError("set must be idx,x,y[,deg]")
    idx = int(parts[0])
    x = float(parts[1])
    y = float(parts[2])
    deg = float(parts[3]) if len(parts) == 4 else 0.0
    return idx, x, y, deg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Apply manual tweaks to one puzzle n in a submission.csv.")
    ap.add_argument("--base", type=Path, default=Path("submission.csv"), help="Input submission.csv")
    ap.add_argument("--out", type=Path, default=Path("submission_tweaked.csv"), help="Output submission.csv")
    ap.add_argument("--n", type=int, required=True, help="Puzzle id (1..200)")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="kaggle",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate used for final validation (kaggle recommended).",
    )
    ap.add_argument("--seed", type=int, default=123, help="Seed used for finalize/repair randomness")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n in the CSV (default: 200)")
    ap.add_argument("--score-total", action="store_true", help="Compute total score delta (slower).")
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="Run the full finalize/repair step after applying tweaks (slower but can fix overlaps).",
    )
    ap.add_argument("--no-write", action="store_true", help="Only report the effect; do not write --out.")

    ap.add_argument(
        "--move",
        action="append",
        default=[],
        help="Relative move: idx,dx,dy[,ddeg]. Can be passed multiple times.",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Absolute set: idx,x,y[,deg]. Can be passed multiple times.",
    )
    ns = ap.parse_args(argv)

    n = int(ns.n)
    if n < 1 or n > 200:
        raise SystemExit("--n must be in [1,200]")

    nmax = int(ns.nmax)
    if nmax < n:
        raise SystemExit("--nmax must be >= --n")
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    base = Path(ns.base).resolve()
    if not base.is_file():
        raise SystemExit(f"base submission not found: {base}")

    overlap_mode: OverlapMode = str(ns.overlap_mode)  # type: ignore[assignment]
    points = np.array(TREE_POINTS, dtype=float)

    puzzles = load_submission(base, nmax=nmax)
    missing = [k for k in range(1, nmax + 1) if k not in puzzles or puzzles[k].shape != (k, 3)]
    if missing:
        raise SystemExit(f"submission missing/invalid puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    poses = np.array(puzzles[n], dtype=float, copy=True)
    s_before = float(packing_score(points, poses))

    # Apply absolute sets first, then relative moves.
    for raw in list(ns.set):
        idx, x, y, deg = _parse_set(str(raw))
        if idx < 0 or idx >= n:
            raise SystemExit(f"--set idx out of range for n={n}: {idx}")
        poses[idx, 0] = float(x)
        poses[idx, 1] = float(y)
        poses[idx, 2] = float(deg)

    for raw in list(ns.move):
        idx, dx, dy, ddeg = _parse_move(str(raw))
        if idx < 0 or idx >= n:
            raise SystemExit(f"--move idx out of range for n={n}: {idx}")
        poses[idx, 0] += float(dx)
        poses[idx, 1] += float(dy)
        poses[idx, 2] = float(np.mod(poses[idx, 2] + float(ddeg), 360.0))

    poses[:, 2] = np.mod(poses[:, 2], 360.0)
    poses_q = quantize_for_submission(fit_xy_in_bounds(poses))

    pair = first_overlap_pair(points, poses_q, mode=overlap_mode)
    if pair is not None and not bool(ns.finalize):
        i, j = pair
        raise SystemExit(
            f"overlap detected after tweak (n={n}, mode={overlap_mode}, pair={i},{j}); rerun with --finalize to attempt repair"
        )

    finalized = poses_q if not bool(ns.finalize) else _finalize_puzzle(points, poses_q, seed=int(ns.seed), puzzle_n=int(n), overlap_mode=overlap_mode)
    pair2 = first_overlap_pair(points, finalized, mode=overlap_mode)
    if pair2 is not None:
        i, j = pair2
        raise SystemExit(f"overlap remains after finalize (n={n}, mode={overlap_mode}, pair={i},{j})")

    s_after = float(packing_score(points, finalized))
    print(f"n={n} s_before={s_before:.12f} s_after={s_after:.12f} delta={s_before - s_after:+.12f}")

    puzzles[n] = np.array(finalized, dtype=float, copy=True)
    out = Path(ns.out).resolve()
    if not bool(ns.no_write):
        _write_submission(out, puzzles, nmax=nmax)
        print(f"wrote: {out}")

    if bool(ns.score_total):
        res0 = score_submission(base, nmax=nmax, check_overlap=True, overlap_mode=overlap_mode, require_complete=True)
        if bool(ns.no_write):
            # Still compute the delta by evaluating the in-memory replacement.
            tmp_path = out
            _write_submission(tmp_path, puzzles, nmax=nmax)
            res1 = score_submission(tmp_path, nmax=nmax, check_overlap=True, overlap_mode=overlap_mode, require_complete=True)
        else:
            res1 = score_submission(out, nmax=nmax, check_overlap=True, overlap_mode=overlap_mode, require_complete=True)
        print(f"total_before={res0.score:.12f}")
        print(f"total_after ={res1.score:.12f}")
        print(f"total_delta ={res0.score - res1.score:+.12f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
