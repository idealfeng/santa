#!/usr/bin/env python3

"""Rank puzzles by estimated empty-space ("slack") and optionally render plots.

Slack metric:
    slack_abs   = s^2 - n * area(tree)
    slack_ratio = 1 - (n * area(tree)) / s^2

This is not a perfect proxy for "visual gaps", but it's a fast way to find
puzzles whose bounding square is relatively under-filled.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.geom_np import packing_score
from santa_packing.scoring import load_submission
from santa_packing.tree_data import TREE_POINTS


def _polygon_area(points: np.ndarray) -> float:
    # Shoelace formula (works for concave simple polygons too).
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass(frozen=True)
class SlackRow:
    n: int
    s: float
    slack_abs: float
    slack_ratio: float


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rank puzzles by slack and optionally render visualizations.")
    ap.add_argument("--submission", type=Path, required=True, help="Path to submission.csv")
    ap.add_argument("--nmin", type=int, default=1, help="Min puzzle n to consider (default 1).")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n to consider (default 200).")
    ap.add_argument("--topk", type=int, default=20, help="How many puzzles to show / render (default 20).")
    ap.add_argument(
        "--metric",
        type=str,
        default="slack_ratio",
        choices=["slack_ratio", "slack_abs"],
        help="Ranking metric (default slack_ratio).",
    )
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional: write full ranking to CSV.")
    ap.add_argument("--render", action="store_true", help="Render plots for the top-k puzzles.")
    ap.add_argument("--outdir", type=Path, default=Path("submissions") / "viz_top_slack", help="Output dir for images.")
    ap.add_argument("--label", action="store_true", help="Label tree indices in rendered plots.")
    ap.add_argument("--highlight-boundary", action="store_true", help="Highlight boundary trees in rendered plots.")
    ap.add_argument("--dpi", type=int, default=180)
    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")
    nmin = int(ns.nmin)
    if nmin < 1 or nmin > nmax:
        raise SystemExit("--nmin must be in [1,nmax]")
    topk = int(ns.topk)
    if topk <= 0:
        raise SystemExit("--topk must be > 0")

    submission = Path(ns.submission).resolve()
    if not submission.is_file():
        raise SystemExit(f"submission not found: {submission}")

    puzzles = load_submission(submission, nmax=nmax)
    points = np.array(TREE_POINTS, dtype=float)
    area_tree = _polygon_area(points)

    rows: list[SlackRow] = []
    for n in range(nmin, nmax + 1):
        poses = puzzles.get(n)
        if poses is None or poses.shape != (n, 3):
            continue
        s = float(packing_score(points, poses))
        box_area = float(s * s)
        slack_abs = box_area - float(n) * area_tree
        slack_ratio = 0.0 if box_area <= 0 else (slack_abs / box_area)
        rows.append(SlackRow(n=n, s=s, slack_abs=slack_abs, slack_ratio=slack_ratio))

    key = (lambda r: r.slack_ratio) if ns.metric == "slack_ratio" else (lambda r: r.slack_abs)
    rows_sorted = sorted(rows, key=key, reverse=True)

    if ns.csv_out is not None:
        out_csv = Path(ns.csv_out).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "s", "slack_abs", "slack_ratio"])
            for r in rows_sorted:
                w.writerow([r.n, f"{r.s:.12f}", f"{r.slack_abs:.12f}", f"{r.slack_ratio:.12f}"])
        print(f"wrote: {out_csv}")

    print(f"tree_area={area_tree:.12f}")
    print(f"top{min(topk, len(rows_sorted))} by {ns.metric}:")
    for r in rows_sorted[:topk]:
        print(f"n={r.n:03d}  s={r.s:.9f}  slack_ratio={r.slack_ratio:.6f}  slack_abs={r.slack_abs:.6f}")

    if ns.render:
        from santa_packing._tools.viz_puzzle import main as viz_main

        outdir = Path(ns.outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        for r in rows_sorted[:topk]:
            out = outdir / f"viz_n{r.n:03d}.png"
            args = [
                "--submission",
                str(submission),
                "--n",
                str(r.n),
                "--out",
                str(out),
                "--dpi",
                str(int(ns.dpi)),
            ]
            if bool(ns.label):
                args.append("--label")
            if bool(ns.highlight_boundary):
                args.append("--highlight-boundary")
            viz_main(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
