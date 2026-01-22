#!/usr/bin/env python3

"""Visualize one puzzle `n` from a `submission.csv`.

This is meant for manual inspection/tweaks:
- Draws every tree polygon for a single puzzle.
- Draws the global AABB and the evaluated bounding square (side length `s`).
- Optionally labels tree indices and highlights boundary trees that define the box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _boundary_indices(bboxes: np.ndarray, *, tol: float) -> tuple[set[int], dict[str, list[int]]]:
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))

    sides: dict[str, list[int]] = {"min_x": [], "max_x": [], "min_y": [], "max_y": []}
    for i in range(bboxes.shape[0]):
        b = bboxes[i]
        if abs(float(b[0]) - min_x) <= tol:
            sides["min_x"].append(i)
        if abs(float(b[2]) - max_x) <= tol:
            sides["max_x"].append(i)
        if abs(float(b[1]) - min_y) <= tol:
            sides["min_y"].append(i)
        if abs(float(b[3]) - max_y) <= tol:
            sides["max_y"].append(i)

    boundary = set(sides["min_x"] + sides["max_x"] + sides["min_y"] + sides["max_y"])
    return boundary, sides


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot one puzzle n from a submission.csv.")
    ap.add_argument("--submission", type=Path, default=Path("submission.csv"))
    ap.add_argument("--n", type=int, required=True, help="Puzzle id (1..200)")
    ap.add_argument("--out", type=Path, default=None, help="Output image path (default: submissions/viz_nNNN.png)")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--pad", type=float, default=0.25, help="Plot padding around the packing (in xy units)")
    ap.add_argument("--label", action="store_true", help="Label tree indices at their centers")
    ap.add_argument("--highlight-boundary", action="store_true", help="Highlight trees touching the global AABB boundary")
    ap.add_argument("--boundary-tol", type=float, default=1e-6, help="Tolerance to classify boundary trees (default: 1e-6)")
    ap.add_argument("--meta-out", type=Path, default=None, help="Optional: write boundary indices to a text file")
    ns = ap.parse_args(argv)

    n = int(ns.n)
    if n < 1 or n > 200:
        raise SystemExit("--n must be in [1,200]")

    submission = Path(ns.submission).resolve()
    if not submission.is_file():
        raise SystemExit(f"submission not found: {submission}")

    out = Path(ns.out) if ns.out is not None else (Path("submissions") / f"viz_n{n:03d}.png")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    from santa_packing.geom_np import packing_bbox, packing_score, polygon_bbox, transform_polygon  # noqa: E402
    from santa_packing.scoring import load_submission  # noqa: E402
    from santa_packing.tree_data import TREE_POINTS  # noqa: E402

    puzzles = load_submission(submission, nmax=n)
    poses = puzzles.get(n)
    if poses is None or poses.shape != (n, 3):
        raise SystemExit(f"puzzle {n} missing or wrong shape: {None if poses is None else poses.shape}")

    points = np.array(TREE_POINTS, dtype=float)
    polys = [transform_polygon(points, pose) for pose in poses]
    bboxes = np.array([polygon_bbox(p) for p in polys], dtype=float)

    bbox = packing_bbox(points, poses)
    min_x, min_y, max_x, max_y = [float(x) for x in bbox]
    w = max_x - min_x
    h = max_y - min_y
    side = float(max(w, h))

    boundary_idxs: set[int] = set()
    sides: dict[str, list[int]] = {"min_x": [], "max_x": [], "min_y": [], "max_y": []}
    if bool(ns.highlight_boundary) or ns.meta_out is not None:
        boundary_idxs, sides = _boundary_indices(bboxes, tol=float(ns.boundary_tol))

    fig, ax = plt.subplots(figsize=(7, 7))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None) or ["C0"]

    for i, p in enumerate(polys):
        c = color_cycle[i % len(color_cycle)]
        lw = 2.0 if (bool(ns.highlight_boundary) and i in boundary_idxs) else 0.8
        ax.plot(p[:, 0], p[:, 1], "-", lw=lw, color=c)
        ax.plot([p[-1, 0], p[0, 0]], [p[-1, 1], p[0, 1]], "-", lw=lw, color=c)
        if bool(ns.label):
            ax.text(float(poses[i, 0]), float(poses[i, 1]), str(i), fontsize=7, ha="center", va="center")

    ax.add_patch(plt.Rectangle((min_x, min_y), w, h, fill=False, lw=2.0, ec="tab:orange", label="AABB"))
    ax.add_patch(plt.Rectangle((min_x, min_y), side, side, fill=False, lw=2.0, ec="tab:red", label="Square"))

    s_val = float(packing_score(points, poses))
    ax.set_aspect("equal", "box")
    ax.set_title(f"n={n}  s={s_val:.9f}  w={w:.9f}  h={h:.9f}  boundary={len(boundary_idxs)}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    pad = float(ns.pad)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

    fig.savefig(out, dpi=int(ns.dpi), bbox_inches="tight")
    print(f"wrote: {out}")

    if ns.meta_out is not None:
        meta_out = Path(ns.meta_out).resolve()
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for key in ("min_x", "max_x", "min_y", "max_y"):
            idxs = ",".join(str(i) for i in sides[key])
            lines.append(f"{key}:{idxs}")
        lines.append(f"all:{','.join(str(i) for i in sorted(boundary_idxs))}")
        meta_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote: {meta_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

