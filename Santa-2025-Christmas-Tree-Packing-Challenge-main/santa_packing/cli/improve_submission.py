#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from santa_packing.geom_np import packing_score, polygon_bbox, polygon_radius, shift_poses_to_origin, transform_polygon
from santa_packing.scoring import OverlapMode, load_submission, polygons_intersect_strict, score_submission
from santa_packing.submission_format import fit_xy_in_bounds, format_submission_value, quantize_for_submission
from santa_packing.tree_data import TREE_POINTS


def _score_from_bboxes(bboxes: np.ndarray) -> float:
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))
    return float(max(max_x - min_x, max_y - min_y))


def _compute_bboxes(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    polys = [transform_polygon(points, pose) for pose in poses]
    return np.array([polygon_bbox(p) for p in polys], dtype=float)


def _radial_subset(poses: np.ndarray, bboxes: np.ndarray, n: int) -> np.ndarray:
    if poses.shape[0] <= n:
        return np.array(poses, dtype=float, copy=True)
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=float)
    d = poses[:, :2] - center[None, :]
    dist2 = np.sum(d * d, axis=1)
    order = np.argsort(dist2, kind="mergesort")
    return np.array(poses[order[:n]], dtype=float, copy=True)


def _greedy_bbox_prune(poses: np.ndarray, bboxes: np.ndarray, n: int) -> np.ndarray:
    m = int(poses.shape[0])
    if m <= n:
        return np.array(poses, dtype=float, copy=True)

    keep = list(range(m))
    eps = 1e-12

    while len(keep) > n:
        bb = bboxes[keep]
        min_x = float(np.min(bb[:, 0]))
        min_y = float(np.min(bb[:, 1]))
        max_x = float(np.max(bb[:, 2]))
        max_y = float(np.max(bb[:, 3]))

        candidates = []
        for idx in keep:
            b = bboxes[idx]
            if (
                abs(float(b[0]) - min_x) <= eps
                or abs(float(b[2]) - max_x) <= eps
                or abs(float(b[1]) - min_y) <= eps
                or abs(float(b[3]) - max_y) <= eps
            ):
                candidates.append(int(idx))

        if not candidates:
            # Fallback: remove the farthest-from-center tree.
            centers = poses[keep, :2]
            center = np.mean(centers, axis=0)
            d = centers - center[None, :]
            dist2 = np.sum(d * d, axis=1)
            rm = keep[int(np.argmax(dist2))]
            keep.remove(rm)
            continue

        best_rm = None
        best_s = None
        for rm in candidates:
            kk = [i for i in keep if i != rm]
            bb2 = bboxes[kk]
            s = _score_from_bboxes(bb2)
            if best_s is None or s < best_s - 1e-12:
                best_s = float(s)
                best_rm = rm
        assert best_rm is not None
        keep.remove(best_rm)

    return np.array(poses[keep], dtype=float, copy=True)


def _smooth_submission(
    points: np.ndarray,
    puzzles: dict[int, np.ndarray],
    *,
    nmax: int,
    window: int,
) -> dict[int, np.ndarray]:
    window = int(window)
    if window <= 0:
        return {n: np.array(p, dtype=float, copy=True) for n, p in puzzles.items()}

    bbox_cache: dict[int, np.ndarray] = {}
    for n in range(1, nmax + 1):
        bbox_cache[n] = _compute_bboxes(points, puzzles[n])

    out: dict[int, np.ndarray] = {n: np.array(p, dtype=float, copy=True) for n, p in puzzles.items()}

    for n in range(1, nmax):
        base = out[n]
        base_s = packing_score(points, base)
        best_s = base_s
        best_pose = base

        for m in range(n + 1, min(nmax, n + window) + 1):
            poses_m = puzzles[m]
            bboxes_m = bbox_cache[m]
            cand1 = _radial_subset(poses_m, bboxes_m, n)
            s1 = packing_score(points, cand1)
            if s1 + 1e-9 < best_s:
                best_s = s1
                best_pose = cand1

            cand2 = _greedy_bbox_prune(poses_m, bboxes_m, n)
            s2 = packing_score(points, cand2)
            if s2 + 1e-9 < best_s:
                best_s = s2
                best_pose = cand2

        if best_s + 1e-9 < base_s:
            out[n] = best_pose

    return out


def _insert_one_tree(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    seed: int,
    center_samples: int,
    angle_samples: int,
    pad_scale: float,
) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    rng = np.random.default_rng(int(seed))

    n = poses.shape[0]
    if n <= 0:
        raise ValueError("Cannot insert into empty packing.")

    polys = [transform_polygon(points, pose) for pose in poses]
    bboxes = np.array([polygon_bbox(p) for p in polys], dtype=float)
    centers = poses[:, :2]

    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))

    rad = float(polygon_radius(points))
    thr2 = (2.0 * rad) ** 2
    pad = float(pad_scale) * 2.0 * rad

    center_samples = int(center_samples)
    angle_samples = int(angle_samples)
    if center_samples <= 0 or angle_samples <= 0:
        raise ValueError("center_samples and angle_samples must be positive.")

    xs = rng.uniform(min_x - pad, max_x + pad, size=(center_samples,))
    ys = rng.uniform(min_y - pad, max_y + pad, size=(center_samples,))

    # Bias half the samples toward the current AABB boundary.
    half = center_samples // 2
    for i in range(half):
        if rng.random() < 0.5:
            xs[i] = (min_x - pad) if rng.random() < 0.5 else (max_x + pad)
            ys[i] = rng.uniform(min_y - pad, max_y + pad)
        else:
            ys[i] = (min_y - pad) if rng.random() < 0.5 else (max_y + pad)
            xs[i] = rng.uniform(min_x - pad, max_x + pad)

    angles = rng.uniform(0.0, 360.0, size=(angle_samples,))

    best_s = float("inf")
    best_pose: np.ndarray | None = None

    for cx, cy in zip(xs, ys):
        for ang in angles:
            cand_pose = np.array([float(cx), float(cy), float(ang)], dtype=float)
            cand_poly = transform_polygon(points, cand_pose)
            cand_center = cand_pose[0:2]

            ok = True
            for j in range(n):
                dx = float(cand_center[0] - centers[j, 0])
                dy = float(cand_center[1] - centers[j, 1])
                if dx * dx + dy * dy > thr2:
                    continue
                if polygons_intersect_strict(cand_poly, polys[j]):
                    ok = False
                    break
            if not ok:
                continue

            cand_bbox = polygon_bbox(cand_poly)
            new_min_x = min(min_x, float(cand_bbox[0]))
            new_min_y = min(min_y, float(cand_bbox[1]))
            new_max_x = max(max_x, float(cand_bbox[2]))
            new_max_y = max(max_y, float(cand_bbox[3]))
            s = max(new_max_x - new_min_x, new_max_y - new_min_y)
            if s < best_s:
                best_s = float(s)
                best_pose = cand_pose

    if best_pose is None:
        raise RuntimeError("Failed to find a feasible insertion.")

    out = np.vstack([poses, best_pose[None, :]])
    return shift_poses_to_origin(points, out)


def _write_submission(path: Path, puzzles: dict[int, np.ndarray], *, nmax: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, nmax + 1):
            poses = puzzles.get(n)
            if poses is None or poses.shape != (n, 3):
                raise ValueError(f"Missing puzzle {n} or wrong shape {None if poses is None else poses.shape}")
            poses = np.array(poses, dtype=float, copy=True)
            poses[:, 2] = np.mod(poses[:, 2], 360.0)
            poses = fit_xy_in_bounds(poses)
            poses = quantize_for_submission(poses)
            for i, (x, y, deg) in enumerate(poses):
                w.writerow(
                    [
                        f"{n:03d}_{i}",
                        format_submission_value(float(x)),
                        format_submission_value(float(y)),
                        format_submission_value(float(deg)),
                    ]
                )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Improve a submission via subset-smoothing + targeted SA.")
    ap.add_argument("submission", type=Path, help="Input submission.csv")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"), help="Output CSV path")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--smooth-window", type=int, default=20, help="Lookahead window for subset smoothing (0 disables).")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="strict",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate used for final validation/repair (strict/kaggle allow touching; conservative counts touching).",
    )

    ap.add_argument("--improve-n200", action="store_true", help="Try to improve puzzle 200 via insert+SA from puzzle 199.")
    ap.add_argument("--n200-insert-seed", type=int, default=123, help="RNG seed for insertion search.")
    ap.add_argument("--n200-insert-centers", type=int, default=4000, help="Candidate centers for insertion.")
    ap.add_argument("--n200-insert-angles", type=int, default=20, help="Angle samples for insertion.")
    ap.add_argument("--n200-insert-pad-scale", type=float, default=0.15, help="Insertion pad scale (in radii).")

    ap.add_argument("--n200-sa-seed", type=int, default=123, help="SA seed for puzzle 200 refinement.")
    ap.add_argument("--n200-sa-batch", type=int, default=32, help="SA batch size for puzzle 200.")
    ap.add_argument("--n200-sa-steps", type=int, default=5000, help="SA steps for puzzle 200.")

    ns = ap.parse_args(argv)
    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    points = np.array(TREE_POINTS, dtype=float)
    base = load_submission(ns.submission, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in base or base[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"Invalid/missing puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    improved = _smooth_submission(points, base, nmax=nmax, window=int(ns.smooth_window))

    if ns.improve_n200 and nmax >= 200:
        try:
            from santa_packing.cli.generate_submission import _run_sa  # noqa: E402
            from santa_packing.cli.generate_submission import _finalize_puzzle  # noqa: E402
        except Exception as exc:
            raise SystemExit(f"JAX SA not available: {exc}")

        poses199 = improved[199]
        inserted = _insert_one_tree(
            points,
            poses199,
            seed=int(ns.n200_insert_seed),
            center_samples=int(ns.n200_insert_centers),
            angle_samples=int(ns.n200_insert_angles),
            pad_scale=float(ns.n200_insert_pad_scale),
        )
        refined = _run_sa(
            200,
            seed=int(ns.n200_sa_seed),
            batch_size=int(ns.n200_sa_batch),
            n_steps=int(ns.n200_sa_steps),
            trans_sigma=0.10,
            rot_sigma=8.0,
            rot_prob=0.15,
            rot_prob_end=0.05,
            swap_prob=0.0,
            swap_prob_end=-1.0,
            push_prob=0.15,
            push_scale=1.0,
            push_square_prob=0.5,
            compact_prob=0.12,
            compact_prob_end=0.25,
            compact_scale=1.0,
            compact_square_prob=0.75,
            teleport_prob=0.05,
            teleport_prob_end=0.01,
            teleport_tries=4,
            teleport_anchor_beta=6.0,
            teleport_ring_mult=1.02,
            teleport_jitter=0.05,
            cooling="geom",
            cooling_power=1.0,
            trans_sigma_nexp=0.0,
            rot_sigma_nexp=0.0,
            sigma_nref=200.0,
            proposal="mixed",
            smart_prob=1.0,
            smart_beta=8.0,
            smart_drift=1.0,
            smart_noise=0.25,
            overlap_lambda=0.0,
            allow_collisions=False,
            initial_poses=inserted,
            objective="packing",
        )
        if refined is None:
            raise SystemExit("SA refinement returned None (is JAX installed for this Python?)")
        # Finalize/repair to guarantee a non-overlapping submission (post-quantization)
        # under the chosen overlap_mode.
        overlap_mode: OverlapMode = str(ns.overlap_mode)  # type: ignore[assignment]
        improved[200] = _finalize_puzzle(
            points,
            refined,
            seed=int(ns.n200_sa_seed) + 13_000,
            puzzle_n=200,
            overlap_mode=overlap_mode,
        )

    _write_submission(ns.out, improved, nmax=nmax)

    # Report local score (no overlap check by default; validate via score_submission CLI + --overlap-mode).
    res = score_submission(ns.out, nmax=nmax, check_overlap=False)
    print(f"wrote: {ns.out}")
    print(f"score(no-overlap): {res.score:.12f}")
    print(f"s_max: {res.s_max:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
