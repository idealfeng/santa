#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from santa_packing.cli.generate_submission import _finalize_puzzle
from santa_packing.cli.improve_submission import _write_submission
from santa_packing.scoring import OverlapMode, load_submission, score_submission
from santa_packing.tree_data import TREE_POINTS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Auto-fix a submission to satisfy a target overlap mode.")
    ap.add_argument("submission", type=Path, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed for repairs (default: 123)")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="conservative",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate to enforce (default: conservative).",
    )
    args = ap.parse_args(argv)

    nmax = int(args.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    base = load_submission(args.submission, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in base or base[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"Invalid/missing puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    points = np.array(TREE_POINTS, dtype=float)
    mode: OverlapMode = str(args.overlap_mode)  # type: ignore[assignment]

    fixed: dict[int, np.ndarray] = {}
    for n in range(1, nmax + 1):
        poses = base[n]
        fixed[n] = _finalize_puzzle(
            points,
            poses,
            seed=int(args.seed) + 1_000_003 * n,
            puzzle_n=n,
            overlap_mode=mode,
        )

    _write_submission(args.out, fixed, nmax=nmax)

    # Report score (without overlap check; validate via score_submission CLI + --overlap-mode).
    res = score_submission(args.out, nmax=nmax, check_overlap=False)
    print(f"wrote: {args.out}")
    print(f"score(no-overlap): {res.score:.12f}")
    print(f"s_max: {res.s_max:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
