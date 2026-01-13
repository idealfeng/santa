#!/usr/bin/env python3

"""Tool to collect SA trajectories as a behavior cloning dataset."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from santa_packing.geom_np import packing_score, polygon_radius, prefix_score, shift_poses_to_origin, transform_polygon
from santa_packing.lattice import lattice_poses
from santa_packing.scoring import polygons_intersect
from santa_packing.tree_data import TREE_POINTS


def _grid_initial(n: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


def _random_initial(n: int, spacing: float, rand_scale: float) -> np.ndarray:
    scale = spacing * math.sqrt(max(n, 1)) * rand_scale
    xy = np.random.uniform(-scale, scale, size=(n, 2))
    theta = np.random.uniform(0.0, 360.0, size=(n, 1))
    return np.concatenate([xy, theta], axis=1)


def _lattice_initial(n: int, *, pattern: str, margin: float, rotate_deg: float) -> np.ndarray:
    return lattice_poses(n, pattern=pattern, margin=margin, rotate_deg=rotate_deg)


def _check_overlaps(points: np.ndarray, poses: np.ndarray) -> bool:
    polys = [transform_polygon(points, pose) for pose in poses]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_intersect(polys[i], polys[j]):
                return True
    return False


def _check_overlap_for_index(points: np.ndarray, poses: np.ndarray, idx: int) -> bool:
    poly_i = transform_polygon(points, poses[idx])
    for j in range(poses.shape[0]):
        if j == idx:
            continue
        if polygons_intersect(poly_i, transform_polygon(points, poses[j])):
            return True
    return False


@dataclass
class RunRecord:
    """Represent a single SA rollout recorded for behavior cloning.

    Attributes:
        poses: Pose snapshots (one per accepted step), each shaped `(n, 3)` as `(x, y, deg)`.
        idxs: Index of the tree moved at each step.
        deltas: Applied deltas `(dx, dy, ddeg)` per step.
        delta_scores: Objective deltas per step (`new_score - old_score`).
        final_score: Final objective value after the rollout.
    """

    poses: List[np.ndarray]
    idxs: List[int]
    deltas: List[np.ndarray]
    delta_scores: List[float]
    final_score: float


def _pose_bboxes(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    bboxes = np.zeros((poses.shape[0], 4), dtype=float)
    for i, pose in enumerate(poses):
        poly = transform_polygon(points, pose)
        min_xy = np.min(poly, axis=0)
        max_xy = np.max(poly, axis=0)
        bboxes[i] = (min_xy[0], min_xy[1], max_xy[0], max_xy[1])
    return bboxes


def _packing_score_from_bboxes(bboxes: np.ndarray) -> float:
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))
    return float(max(max_x - min_x, max_y - min_y))


def _prefix_packing_score(points: np.ndarray, poses: np.ndarray) -> float:
    bboxes = _pose_bboxes(points, poses)
    min_x = float(bboxes[0, 0])
    min_y = float(bboxes[0, 1])
    max_x = float(bboxes[0, 2])
    max_y = float(bboxes[0, 3])
    s_vals = np.zeros((poses.shape[0],), dtype=float)
    s_vals[0] = max(max_x - min_x, max_y - min_y)
    for i in range(1, poses.shape[0]):
        min_x = min(min_x, float(bboxes[i, 0]))
        min_y = min(min_y, float(bboxes[i, 1]))
        max_x = max(max_x, float(bboxes[i, 2]))
        max_y = max(max_y, float(bboxes[i, 3]))
        s_vals[i] = max(max_x - min_x, max_y - min_y)
    return float(prefix_score(s_vals))


def _run_sa_collect(
    n: int,
    *,
    steps: int,
    t_start: float,
    t_end: float,
    trans_sigma: float,
    rot_sigma: float,
    init_mode: str,
    rand_scale: float,
    lattice_pattern: str,
    lattice_margin: float,
    lattice_rotate: float,
    objective: str,
    points: np.ndarray,
) -> RunRecord:
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2
    if init_mode == "grid":
        poses = _grid_initial(n, spacing)
    elif init_mode == "random":
        poses = _random_initial(n, spacing, rand_scale)
    elif init_mode == "lattice":
        poses = _lattice_initial(n, pattern=lattice_pattern, margin=lattice_margin, rotate_deg=lattice_rotate)
    else:
        poses = _grid_initial(n, spacing)

    poses = np.array(poses, dtype=float)
    poses = shift_poses_to_origin(points, poses)
    if _check_overlaps(points, poses):
        poses = shift_poses_to_origin(points, _grid_initial(n, spacing))
    if objective == "prefix":
        score = _prefix_packing_score(points, poses)
    else:
        score = packing_score(points, poses)

    accepted_poses: List[np.ndarray] = []
    accepted_idxs: List[int] = []
    accepted_deltas: List[np.ndarray] = []
    accepted_delta_scores: List[float] = []

    for i in range(steps):
        frac = i / max(steps, 1)
        temp = t_start * (t_end / t_start) ** frac
        idx = np.random.randint(0, n)
        delta = np.random.normal(size=(3,))
        delta[0:2] *= trans_sigma * temp
        delta[2] *= rot_sigma * temp
        candidate = poses.copy()
        candidate[idx] = candidate[idx] + delta
        candidate[idx, 2] = np.mod(candidate[idx, 2], 360.0)

        if _check_overlap_for_index(points, candidate, idx):
            continue

        if objective == "prefix":
            cand_score = _prefix_packing_score(points, candidate)
        else:
            cand_score = packing_score(points, candidate)
        dscore = cand_score - score
        if dscore < 0 or np.random.rand() < math.exp(-dscore / max(temp, 1e-9)):
            accepted_poses.append(poses.copy())
            accepted_idxs.append(idx)
            accepted_deltas.append(delta.copy())
            accepted_delta_scores.append(float(score - cand_score))
            poses = candidate
            score = cand_score

    return RunRecord(accepted_poses, accepted_idxs, accepted_deltas, accepted_delta_scores, score)


def main(argv: list[str] | None = None) -> int:
    """Run SA rollouts, record accepted moves, and save them to a `.npz` file."""
    ap = argparse.ArgumentParser(description="Collect behavior cloning dataset from SA runs")
    ap.add_argument("--n-list", type=str, default="25,50,100", help="Comma-separated Ns")
    ap.add_argument("--runs-per-n", type=int, default=5, help="SA runs per N")
    ap.add_argument("--steps", type=int, default=400, help="SA steps per run")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (numpy)")
    ap.add_argument("--t-start", type=float, default=1.0)
    ap.add_argument("--t-end", type=float, default=0.001)
    ap.add_argument("--trans-sigma", type=float, default=0.2)
    ap.add_argument("--rot-sigma", type=float, default=10.0)
    ap.add_argument("--objective", type=str, default="packing", choices=["packing", "prefix"])
    ap.add_argument("--init", type=str, default="grid", choices=["grid", "random", "mix", "lattice", "all"])
    ap.add_argument("--rand-scale", type=float, default=0.3)
    ap.add_argument("--lattice-pattern", type=str, default="hex", choices=["hex", "square"])
    ap.add_argument("--lattice-margin", type=float, default=0.02)
    ap.add_argument("--lattice-rotate", type=float, default=0.0)
    ap.add_argument("--best-only", action="store_true", help="Keep only best run per N")
    ap.add_argument("--out", type=Path, default=Path("runs") / "sa_bc_dataset.npz")
    args = ap.parse_args(argv)

    np.random.seed(int(args.seed))
    ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    points = np.array(TREE_POINTS, dtype=float)

    payload: Dict[str, np.ndarray] = {}
    for n in ns:
        runs: List[RunRecord] = []
        for r in range(args.runs_per_n):
            if args.init == "mix":
                init_mode = "grid" if r % 2 == 0 else "random"
            elif args.init == "all":
                init_mode = ["grid", "random", "lattice"][r % 3]
            else:
                init_mode = args.init
            run = _run_sa_collect(
                n,
                steps=args.steps,
                t_start=args.t_start,
                t_end=args.t_end,
                trans_sigma=args.trans_sigma,
                rot_sigma=args.rot_sigma,
                init_mode=init_mode,
                rand_scale=args.rand_scale,
                lattice_pattern=args.lattice_pattern,
                lattice_margin=args.lattice_margin,
                lattice_rotate=args.lattice_rotate,
                objective=args.objective,
                points=points,
            )
            runs.append(run)

        if args.best_only and runs:
            runs = [min(runs, key=lambda r: r.final_score)]

        poses = np.concatenate([np.array(r.poses) for r in runs], axis=0) if runs else np.zeros((0, n, 3))
        idxs = (
            np.concatenate([np.array(r.idxs, dtype=int) for r in runs], axis=0) if runs else np.zeros((0,), dtype=int)
        )
        deltas = np.concatenate([np.array(r.deltas, dtype=float) for r in runs], axis=0) if runs else np.zeros((0, 3))
        dscores = (
            np.concatenate([np.array(r.delta_scores, dtype=float) for r in runs], axis=0)
            if runs
            else np.zeros((0,), dtype=float)
        )

        payload[f"poses_n{n}"] = poses
        payload[f"idx_n{n}"] = idxs
        payload[f"delta_n{n}"] = deltas
        payload[f"dscore_n{n}"] = dscores

        print(f"N={n} samples={poses.shape[0]} best_only={args.best_only}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(f"Saved dataset to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
