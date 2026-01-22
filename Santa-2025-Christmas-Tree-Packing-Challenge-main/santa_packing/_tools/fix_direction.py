#!/usr/bin/env python3
"""Rigid-rotate each puzzle to minimize its axis-aligned bounding square.

This is a local port of the common Kaggle "fix direction" trick:
- For each puzzle n, rotate the entire packing by a single angle in [0, 90)
  (rotation around the packing center + add angle to each tree's `deg`).
- Accept the rotation only if it reduces `s_n` by at least a small epsilon.

The operation is a rigid transform, so it preserves non-overlap (up to
floating-point roundoff).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_bbox
from santa_packing.scoring import load_submission
from santa_packing.tree_data import TREE_POINTS


def _transform_points(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Return all transformed vertices for the packing as a `(N*V, 2)` array."""
    pts = np.array(points, dtype=float)
    poses = np.array(poses, dtype=float)
    x = poses[:, 0]
    y = poses[:, 1]
    theta = np.deg2rad(poses[:, 2])
    c = np.cos(theta)
    s = np.sin(theta)
    px = pts[:, 0]
    py = pts[:, 1]
    # (N,V): rotate template then translate by x/y.
    X = (c[:, None] * px[None, :]) - (s[:, None] * py[None, :]) + x[:, None]
    Y = (s[:, None] * px[None, :]) + (c[:, None] * py[None, :]) + y[:, None]
    return np.stack([X, Y], axis=-1).reshape((-1, 2))


def _bbox_side_for_angle(points_xy: np.ndarray, angle_deg: float) -> float:
    """AABB square side length after rotating `points_xy` by `angle_deg` around origin."""
    angle = math.radians(float(angle_deg))
    c = math.cos(angle)
    s = math.sin(angle)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    rx = c * x - s * y
    ry = s * x + c * y
    w = float(np.max(rx) - np.min(rx))
    h = float(np.max(ry) - np.min(ry))
    return float(w if w >= h else h)


def _best_rotation_angle(
    points_xy: np.ndarray,
    *,
    angle_min: float,
    angle_max: float,
    coarse_step: float,
    refine_iters: int,
) -> tuple[float, float]:
    """Return (best_angle_deg, best_side)."""
    angle_min = float(angle_min)
    angle_max = float(angle_max)
    coarse_step = float(coarse_step)
    if angle_max <= angle_min:
        raise ValueError("angle_max must be > angle_min")
    if coarse_step <= 0:
        raise ValueError("coarse_step must be > 0")
    refine_iters = int(refine_iters)
    if refine_iters < 0:
        raise ValueError("refine_iters must be >= 0")

    # Coarse grid search (cheap, robust against non-unimodality).
    angles = np.arange(angle_min, angle_max + 0.5 * coarse_step, coarse_step, dtype=float)
    best_angle = float(angles[0])
    best_side = float("inf")
    for a in angles:
        s = _bbox_side_for_angle(points_xy, float(a))
        if s < best_side:
            best_side = s
            best_angle = float(a)

    # Local refine (golden section) around the best coarse angle.
    if refine_iters == 0:
        return best_angle, best_side

    lo = max(angle_min, best_angle - coarse_step)
    hi = min(angle_max, best_angle + coarse_step)
    if hi - lo <= 1e-12:
        return best_angle, best_side

    phi = (math.sqrt(5.0) - 1.0) * 0.5  # 0.618...
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1 = _bbox_side_for_angle(points_xy, x1)
    f2 = _bbox_side_for_angle(points_xy, x2)

    for _ in range(refine_iters):
        if f1 <= f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - phi * (hi - lo)
            f1 = _bbox_side_for_angle(points_xy, x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + phi * (hi - lo)
            f2 = _bbox_side_for_angle(points_xy, x2)

    best_angle = float((lo + hi) * 0.5)
    best_side = _bbox_side_for_angle(points_xy, best_angle)
    return best_angle, float(best_side)


def _rotate_poses_about_xy(poses: np.ndarray, *, center_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses
    angle = math.radians(float(angle_deg))
    c = math.cos(angle)
    s = math.sin(angle)
    rot_T = np.array([[c, s], [-s, c]], dtype=float)
    xy = poses[:, 0:2]
    shifted = xy - center_xy[None, :]
    poses[:, 0:2] = shifted @ rot_T + center_xy[None, :]
    poses[:, 2] = np.mod(poses[:, 2] + float(angle_deg), 360.0)
    return poses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rigid-rotate each puzzle to shrink its AABB square (fix-direction).")
    ap.add_argument("--submission", type=Path, default=Path("submission.csv"), help="Input submission.csv")
    ap.add_argument("--out", type=Path, default=Path("submission_fix_direction.csv"), help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--min-improvement", type=float, default=1e-5, help="Minimum side-length improvement to accept.")
    ap.add_argument("--angle-min", type=float, default=0.001, help="Search min angle (deg).")
    ap.add_argument("--angle-max", type=float, default=89.999, help="Search max angle (deg).")
    ap.add_argument("--coarse-step", type=float, default=1.0, help="Coarse scan step in degrees (default: 1.0).")
    ap.add_argument("--refine-iters", type=int, default=20, help="Golden-section refinement iterations (default: 20).")
    ap.add_argument("--verbose", action="store_true", help="Print per-n improvements.")
    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    points = np.array(TREE_POINTS, dtype=float)
    puzzles = load_submission(ns.submission, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in puzzles or puzzles[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"Invalid/missing puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    out: dict[int, np.ndarray] = {n: np.array(puzzles[n], dtype=float, copy=True) for n in range(1, nmax + 1)}
    improved = 0

    for n in range(nmax, 1, -1):
        poses = out[n]
        pts_xy = _transform_points(points, poses)

        side0 = _bbox_side_for_angle(pts_xy, 0.0)
        best_angle, best_side = _best_rotation_angle(
            pts_xy,
            angle_min=float(ns.angle_min),
            angle_max=float(ns.angle_max),
            coarse_step=float(ns.coarse_step),
            refine_iters=int(ns.refine_iters),
        )
        if side0 - best_side <= float(ns.min_improvement):
            continue

        bbox = packing_bbox(points, poses)
        center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=float)
        cand = _rotate_poses_about_xy(poses, center_xy=center, angle_deg=float(best_angle))
        out[n] = cand
        improved += 1
        if ns.verbose:
            print(f"[n={n:03d}] side {side0:.9f} -> {best_side:.9f}  angle={best_angle:.6f}")

    _write_submission(ns.out, out, nmax=nmax)
    if ns.verbose:
        print(f"wrote: {ns.out} (improved_puzzles={improved})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

