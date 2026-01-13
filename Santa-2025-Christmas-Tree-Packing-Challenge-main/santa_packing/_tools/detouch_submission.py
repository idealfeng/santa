#!/usr/bin/env python3

"""Tool to post-process a submission by "detouching" each puzzle.

Some evaluators treat boundary touching as overlap; a small uniform expansion of
each puzzle around its centroid can resolve contact-only collisions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from santa_packing.scoring import load_submission
from santa_packing.submission_format import fit_xy_in_bounds, format_submission_value, quantize_for_submission


def _scale_about_centroid(poses: np.ndarray, *, scale: float) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses
    center = np.mean(poses[:, 0:2], axis=0)
    poses[:, 0:2] = center[None, :] + (poses[:, 0:2] - center[None, :]) * float(scale)
    return poses


def main(argv: list[str] | None = None) -> int:
    """Read a submission CSV, apply scaling, and write a new submission CSV."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description="Detouch a submission by uniformly scaling each puzzle about its centroid."
    )
    ap.add_argument("input", type=Path, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--scale", type=float, default=1.01, help="Uniform XY scale factor (>1 pushes trees apart).")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n to process (default 200).")
    args = ap.parse_args(argv)

    if args.scale <= 0.0:
        raise SystemExit("--scale must be > 0")

    puzzles = load_submission(args.input, nmax=int(args.nmax))
    if not puzzles:
        raise SystemExit(f"No puzzles found in {args.input}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])
        for n in range(1, int(args.nmax) + 1):
            poses = puzzles.get(n)
            if poses is None or poses.shape != (n, 3):
                raise SystemExit(f"Missing puzzle {n} or wrong shape (got {None if poses is None else poses.shape})")
            poses = _scale_about_centroid(poses, scale=float(args.scale))
            poses = fit_xy_in_bounds(poses)
            poses = quantize_for_submission(poses)
            for i, (x, y, deg) in enumerate(poses):
                writer.writerow(
                    [
                        f"{n:03d}_{i}",
                        format_submission_value(x),
                        format_submission_value(y),
                        format_submission_value(deg),
                    ]
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
