"""NumPy post-optimization operators and repair utilities.

These routines are used by the submission generator and sweep/ensemble tooling:
- overlap detection/repair (best-effort)
- deterministic hill-climb refinements
- simple GA and LNS-style metaheuristics
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import EPS
from .geom_np import polygon_bbox, polygon_radius, shift_poses_to_origin, transform_polygon
from .scoring import (
    KAGGLE_CLEARANCE,
    OverlapMode,
    polygons_intersect as polygons_intersect_conservative,
    polygons_intersect_strict,
)


def _packing_score_from_bboxes(bboxes: np.ndarray) -> float:
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))
    return float(max(max_x - min_x, max_y - min_y))


def _aabb_overlaps(a: np.ndarray, b: np.ndarray, *, eps: float = EPS) -> bool:
    """Return True if AABBs overlap (treating touch within eps as overlap candidate)."""
    return not (
        float(a[2]) < float(b[0]) - float(eps)
        or float(b[2]) < float(a[0]) - float(eps)
        or float(a[3]) < float(b[1]) - float(eps)
        or float(b[3]) < float(a[1]) - float(eps)
    )


def _intersects_for_mode(mode: OverlapMode):
    if mode in {"strict", "kaggle"}:
        return polygons_intersect_strict
    if mode == "conservative":
        return polygons_intersect_conservative
    raise ValueError(f"Unknown overlap mode: {mode!r}")


def _has_overlaps(points: np.ndarray, poses: np.ndarray, *, mode: OverlapMode) -> bool:
    poses = np.array(poses, dtype=float, copy=False)
    if poses.shape[0] <= 1:
        return False
    intersects = _intersects_for_mode(mode)

    centers = poses[:, :2]
    clearance = float(KAGGLE_CLEARANCE) if mode == "kaggle" else 0.0
    rad = float(polygon_radius(points)) + clearance
    dist_thr = 2.0 * rad + float(EPS)
    thr2 = dist_thr * dist_thr

    polys = [transform_polygon(points, pose) for pose in poses]
    n = poses.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(centers[i, 0] - centers[j, 0])
            dy = float(centers[i, 1] - centers[j, 1])
            if dx * dx + dy * dy > thr2:
                continue
            if intersects(polys[i], polys[j]):
                return True
    return False


def has_overlaps(points: np.ndarray, poses: np.ndarray, *, overlap_mode: OverlapMode = "strict") -> bool:
    """Return True if any pair of polygons overlaps (NumPy predicate).

    Args:
        points: Local polygon vertices `(V, 2)`.
        poses: Poses `(N, 3)` as `[x, y, theta_deg]`.
        overlap_mode: Overlap predicate (`strict` allows touching; `conservative` counts touching).

    Returns:
        True if any overlap is detected.
    """
    return _has_overlaps(points, poses, mode=overlap_mode)


def repair_overlaps(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    seed: int = 1,
    max_iters: int = 2000,
    step_xy: float = 0.01,
    step_deg: float = 0.0,
    overlap_mode: OverlapMode = "strict",
) -> np.ndarray | None:
    """Best-effort repair for colliding packings (translation nudges).

    Args:
        points: Local polygon vertices `(V, 2)`.
        poses: Initial poses `(N, 3)`.
        seed: RNG seed.
        max_iters: Repair iteration budget.
        step_xy: Translation step size.
        step_deg: Rotation step size (0 disables rotation during repair).
        overlap_mode: Overlap predicate (`strict` allows touching; `conservative` counts touching).

    Returns:
        Repaired poses (shifted to origin) if feasible, otherwise None.
    """
    rng = np.random.default_rng(int(seed))
    return _repair_overlaps(
        points,
        poses,
        rng=rng,
        max_iters=int(max_iters),
        step_xy=float(step_xy),
        step_deg=float(step_deg),
        overlap_mode=overlap_mode,
    )


@dataclass
class _PackingState:
    poses: np.ndarray
    centers: np.ndarray
    polys: list[np.ndarray]
    bboxes: np.ndarray
    score: float
    thr2: float


def _build_state(points: np.ndarray, poses: np.ndarray) -> _PackingState:
    poses = np.array(poses, dtype=float, copy=True)
    polys = [transform_polygon(points, pose) for pose in poses]
    bboxes = np.stack([polygon_bbox(p) for p in polys], axis=0)
    score = _packing_score_from_bboxes(bboxes)
    rad = float(polygon_radius(points))
    dist_thr = 2.0 * rad + float(EPS)
    thr2 = dist_thr * dist_thr
    return _PackingState(
        poses=poses,
        centers=poses[:, :2].copy(),
        polys=polys,
        bboxes=bboxes,
        score=score,
        thr2=thr2,
    )


def _state_has_overlaps(state: _PackingState, *, overlap_mode: OverlapMode) -> bool:
    intersects = _intersects_for_mode(overlap_mode)
    clearance = float(KAGGLE_CLEARANCE) if overlap_mode == "kaggle" else 0.0
    thr2 = state.thr2
    if clearance > 0.0:
        thr = math.sqrt(float(state.thr2))
        thr2 = (thr + 2.0 * clearance) ** 2
    n = state.poses.shape[0]
    if n <= 1:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(state.centers[i, 0] - state.centers[j, 0])
            dy = float(state.centers[i, 1] - state.centers[j, 1])
            if dx * dx + dy * dy > thr2:
                continue
            if intersects(state.polys[i], state.polys[j]):
                return True
    return False


def _collides_one_vs_all(
    state: _PackingState,
    idx: int,
    cand_center: np.ndarray,
    cand_poly: np.ndarray,
    *,
    overlap_mode: OverlapMode,
) -> bool:
    intersects = _intersects_for_mode(overlap_mode)
    cand_bbox = polygon_bbox(cand_poly)
    clearance = float(KAGGLE_CLEARANCE) if overlap_mode == "kaggle" else 0.0
    thr2 = state.thr2
    if clearance > 0.0:
        thr = math.sqrt(float(state.thr2))
        thr2 = (thr + 2.0 * clearance) ** 2
    n = state.poses.shape[0]
    for j in range(n):
        if j == idx:
            continue
        dx = float(cand_center[0] - state.centers[j, 0])
        dy = float(cand_center[1] - state.centers[j, 1])
        if dx * dx + dy * dy > thr2:
            continue
        if not _aabb_overlaps(cand_bbox, state.bboxes[j], eps=float(EPS) + clearance):
            continue
        if intersects(cand_poly, state.polys[j]):
            return True
    return False


def hill_climb(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    step_xy: float = 0.01,
    step_deg: float = 2.0,
    max_passes: int = 2,
    overlap_mode: OverlapMode = "strict",
    tol: float = 1e-12,
) -> np.ndarray:
    """Deterministic local search over single-tree moves.

    For each tree, tries a small set of axis-aligned translation and rotation
    deltas and accepts only strict improvements in packing score.

    Args:
        points: Local polygon vertices `(V, 2)`.
        poses: Initial poses `(N, 3)`.
        step_xy: Translation step size.
        step_deg: Rotation step size.
        max_passes: Maximum full passes over all trees.
        tol: Improvement tolerance.

    Returns:
        Improved poses shifted to the origin.
    """

    state = _build_state(points, poses)
    n = state.poses.shape[0]
    if n <= 1:
        return shift_poses_to_origin(points, state.poses)

    moves: list[tuple[float, float, float]] = [
        (step_xy, 0.0, 0.0),
        (-step_xy, 0.0, 0.0),
        (0.0, step_xy, 0.0),
        (0.0, -step_xy, 0.0),
        (0.0, 0.0, step_deg),
        (0.0, 0.0, -step_deg),
    ]

    for _pass in range(max_passes):
        improved_pass = False
        for idx in range(n):
            best_score = state.score
            best_pose: np.ndarray | None = None
            best_poly: np.ndarray | None = None
            best_bbox: np.ndarray | None = None

            base_pose = state.poses[idx].copy()
            for dx, dy, ddeg in moves:
                cand_pose = base_pose.copy()
                cand_pose[0] += dx
                cand_pose[1] += dy
                cand_pose[2] = float(math.fmod(cand_pose[2] + ddeg, 360.0))
                if cand_pose[2] < 0.0:
                    cand_pose[2] += 360.0

                cand_poly = transform_polygon(points, cand_pose)
                cand_center = cand_pose[:2]
                if _collides_one_vs_all(state, idx, cand_center, cand_poly, overlap_mode=overlap_mode):
                    continue

                cand_bbox = polygon_bbox(cand_poly)
                tmp = state.bboxes.copy()
                tmp[idx] = cand_bbox
                cand_score = _packing_score_from_bboxes(tmp)

                if cand_score + tol < best_score:
                    best_score = cand_score
                    best_pose = cand_pose
                    best_poly = cand_poly
                    best_bbox = cand_bbox

            if best_pose is not None:
                state.poses[idx] = best_pose
                state.centers[idx] = best_pose[:2]
                state.polys[idx] = best_poly  # type: ignore[assignment]
                state.bboxes[idx] = best_bbox  # type: ignore[assignment]
                state.score = best_score
                improved_pass = True

        if not improved_pass:
            break

    return shift_poses_to_origin(points, state.poses)


def hill_climb_boundary(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    steps: int = 50,
    step_xy: float = 0.01,
    step_deg: float = 0.0,
    seed: int = 1,
    candidates: int = 8,
    overlap_mode: OverlapMode = "strict",
    tol: float = 1e-12,
) -> np.ndarray:
    """Short boundary-focused hill-climb to reduce the packing bounding square.

    Unlike `hill_climb` (which scans every tree), this runs a small fixed budget of
    steps and targets trees near the current AABB boundary.

    Args:
        points: Local polygon vertices `(V, 2)`.
        poses: Initial poses `(N, 3)`.
        steps: Number of iterations.
        step_xy: Translation step size.
        step_deg: Rotation step size.
        seed: RNG seed.
        candidates: Approximate number of boundary candidates per step.
        tol: Improvement tolerance.

    Returns:
        Improved poses shifted to the origin.
    """

    state = _build_state(points, poses)
    n = state.poses.shape[0]
    if n <= 1 or steps <= 0:
        return shift_poses_to_origin(points, state.poses)

    rng = np.random.default_rng(int(seed))

    for _ in range(int(steps)):
        min_x = float(np.min(state.bboxes[:, 0]))
        min_y = float(np.min(state.bboxes[:, 1]))
        max_x = float(np.max(state.bboxes[:, 2]))
        max_y = float(np.max(state.bboxes[:, 3]))
        center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=float)

        idxs = np.arange(n, dtype=np.int32)
        if candidates > 0 and n > candidates:
            # Sample a few near-boundary trees (weighted); keeps runtime stable for larger n.
            chosen: list[int] = []
            for _ in range(int(candidates)):
                idx, _axis, _center = _pick_boundary_tree(state.bboxes, rng=rng, k=8)
                chosen.append(int(idx))
            idxs = np.array(sorted(set(chosen)), dtype=np.int32)

        best_score = state.score
        best_idx: int | None = None
        best_pose: np.ndarray | None = None
        best_poly: np.ndarray | None = None
        best_bbox: np.ndarray | None = None

        for idx in idxs:
            base_pose = state.poses[int(idx)].copy()
            dir_xy = center - base_pose[:2]
            sx = 0.0 if abs(float(dir_xy[0])) < 1e-12 else float(math.copysign(step_xy, float(dir_xy[0])))
            sy = 0.0 if abs(float(dir_xy[1])) < 1e-12 else float(math.copysign(step_xy, float(dir_xy[1])))

            moves: list[tuple[float, float, float]] = [
                (sx, 0.0, 0.0),
                (0.0, sy, 0.0),
                (sx, sy, 0.0),
            ]
            if step_deg != 0.0:
                moves.append((0.0, 0.0, float(step_deg)))
                moves.append((0.0, 0.0, float(-step_deg)))

            for dx, dy, ddeg in moves:
                if dx == 0.0 and dy == 0.0 and ddeg == 0.0:
                    continue
                cand_pose = base_pose.copy()
                cand_pose[0] += dx
                cand_pose[1] += dy
                if ddeg != 0.0:
                    cand_pose[2] = float(math.fmod(cand_pose[2] + ddeg, 360.0))
                    if cand_pose[2] < 0.0:
                        cand_pose[2] += 360.0

                cand_poly = transform_polygon(points, cand_pose)
                cand_center = cand_pose[:2]
                if _collides_one_vs_all(state, int(idx), cand_center, cand_poly, overlap_mode=overlap_mode):
                    continue

                cand_bbox = polygon_bbox(cand_poly)
                tmp = state.bboxes.copy()
                tmp[int(idx)] = cand_bbox
                cand_score = _packing_score_from_bboxes(tmp)
                if cand_score + tol < best_score:
                    best_score = cand_score
                    best_idx = int(idx)
                    best_pose = cand_pose
                    best_poly = cand_poly
                    best_bbox = cand_bbox

        if best_idx is None:
            break

        state.poses[best_idx] = best_pose  # type: ignore[assignment]
        state.centers[best_idx] = best_pose[:2]  # type: ignore[index]
        state.polys[best_idx] = best_poly  # type: ignore[assignment]
        state.bboxes[best_idx] = best_bbox  # type: ignore[assignment]
        state.score = best_score

    return shift_poses_to_origin(points, state.poses)


def _pick_boundary_tree(
    bboxes: np.ndarray,
    *,
    rng: np.random.Generator,
    k: int = 8,
) -> tuple[int, int, np.ndarray]:
    """Pick a tree near the current packing AABB boundary.

    Returns: (idx, axis, center_xy) where axis is 0 (x) or 1 (y).
    """
    min_x = float(np.min(bboxes[:, 0]))
    min_y = float(np.min(bboxes[:, 1]))
    max_x = float(np.max(bboxes[:, 2]))
    max_y = float(np.max(bboxes[:, 3]))
    width = max_x - min_x
    height = max_y - min_y
    axis = 0 if width >= height else 1
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=float)

    if axis == 0:
        slack = np.minimum(bboxes[:, 0] - min_x, max_x - bboxes[:, 2])
        scale = max(width, 1e-9)
    else:
        slack = np.minimum(bboxes[:, 1] - min_y, max_y - bboxes[:, 3])
        scale = max(height, 1e-9)
    slack = np.maximum(slack, 0.0)

    # Sample from the k most boundary-ish trees to avoid always picking the same index.
    order = np.argsort(slack, kind="mergesort")
    topk = order[: max(1, min(int(k), order.shape[0]))]
    weights = np.exp(-(slack[topk] / scale) * 8.0)
    weights = weights / np.sum(weights)
    idx = int(rng.choice(topk, p=weights))
    return idx, axis, center


def _pick_boundary_tree_from_centers(
    centers: np.ndarray,
    *,
    rng: np.random.Generator,
    k: int = 8,
) -> tuple[int, int, np.ndarray]:
    min_x = float(np.min(centers[:, 0]))
    min_y = float(np.min(centers[:, 1]))
    max_x = float(np.max(centers[:, 0]))
    max_y = float(np.max(centers[:, 1]))
    width = max_x - min_x
    height = max_y - min_y
    axis = 0 if width >= height else 1
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=float)

    if axis == 0:
        slack = np.minimum(centers[:, 0] - min_x, max_x - centers[:, 0])
        scale = max(width, 1e-9)
    else:
        slack = np.minimum(centers[:, 1] - min_y, max_y - centers[:, 1])
        scale = max(height, 1e-9)
    slack = np.maximum(slack, 0.0)

    order = np.argsort(slack, kind="mergesort")
    topk = order[: max(1, min(int(k), order.shape[0]))]
    weights = np.exp(-(slack[topk] / scale) * 8.0)
    weights = weights / np.sum(weights)
    idx = int(rng.choice(topk, p=weights))
    return idx, axis, center


def _repair_overlaps(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    rng: np.random.Generator,
    max_iters: int = 200,
    step_xy: float = 0.01,
    step_deg: float = 0.0,
    overlap_mode: OverlapMode = "strict",
) -> np.ndarray | None:
    """Try to repair overlaps by nudging colliding trees apart."""
    state = _build_state(points, poses)
    n = state.poses.shape[0]
    if n <= 1:
        return shift_poses_to_origin(points, state.poses)

    intersects = _intersects_for_mode(overlap_mode)
    dist_thr = 2.0 * float(polygon_radius(points)) + float(EPS)
    thr2 = float(dist_thr * dist_thr)

    def _aabb_overlaps(a: np.ndarray, b: np.ndarray) -> bool:
        return not (
            float(a[2]) < float(b[0]) - float(EPS)
            or float(b[2]) < float(a[0]) - float(EPS)
            or float(a[3]) < float(b[1]) - float(EPS)
            or float(b[3]) < float(a[1]) - float(EPS)
        )

    def _grid_candidate_pairs(centers: np.ndarray, *, cell_size: float) -> list[tuple[int, int]]:
        grid: dict[tuple[int, int], list[int]] = {}
        inv = 1.0 / float(max(cell_size, 1e-12))
        for idx, (x, y) in enumerate(centers):
            gx = int(math.floor(float(x) * inv))
            gy = int(math.floor(float(y) * inv))
            grid.setdefault((gx, gy), []).append(int(idx))

        pairs: set[tuple[int, int]] = set()
        for (gx, gy), idxs in grid.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = grid.get((gx + dx, gy + dy))
                    if not other:
                        continue
                    for i in idxs:
                        for j in other:
                            if j <= i:
                                continue
                            pairs.add((i, j))
        # Do not sort: sorting is O(P log P) and can be pathological when the
        # first colliding pair is late in lexicographic order (we'd scan many
        # non-colliding pairs every iteration). Randomized order finds a
        # colliding pair quickly on dense/touch-heavy packings.
        return list(pairs)

    for _ in range(max_iters):
        pair: tuple[int, int] | None = None
        cand_pairs = _grid_candidate_pairs(state.centers, cell_size=dist_thr)
        if len(cand_pairs) > 1:
            rng.shuffle(cand_pairs)
        for i, j in cand_pairs:
            dx = float(state.centers[i, 0] - state.centers[j, 0])
            dy = float(state.centers[i, 1] - state.centers[j, 1])
            if dx * dx + dy * dy > thr2:
                continue
            if not _aabb_overlaps(state.bboxes[i], state.bboxes[j]):
                continue
            if intersects(state.polys[i], state.polys[j]):
                pair = (i, j)
                break
        if pair is None:
            return shift_poses_to_origin(points, state.poses)

        i, j = pair

        moved = False
        move_order = [i, j]
        if rng.random() < 0.5:
            move_order.reverse()

        for move in move_order:
            other = j if move == i else i
            base_pose = state.poses[move].copy()

            direction = state.centers[move] - state.centers[other]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                ang = float(rng.uniform(0.0, 2.0 * math.pi))
                direction = np.array([math.cos(ang), math.sin(ang)], dtype=float)
                norm = 1.0
            unit = direction / norm
            perp = np.array([-unit[1], unit[0]], dtype=float)

            # Prefer a small deterministic move set first: more reliable for
            # "sliding touch" edge contacts than pure random jitter, while still
            # keeping the per-iteration budget bounded.
            dirs: list[np.ndarray] = [
                unit,
                perp,
                -perp,
                unit + perp,
                unit - perp,
            ]
            normed_dirs: list[np.ndarray] = []
            for d in dirs:
                d_norm = float(np.linalg.norm(d))
                if d_norm < 1e-12:
                    continue
                normed_dirs.append(d / d_norm)

            if step_deg != 0.0:
                rot = float(step_deg)
                # Rotation-only tries first: often breaks collinear contacts
                # without expanding the packing.
                for ddeg in (rot, -rot):
                    cand_pose = base_pose.copy()
                    cand_pose[2] = float(math.fmod(cand_pose[2] + ddeg, 360.0))
                    if cand_pose[2] < 0.0:
                        cand_pose[2] += 360.0
                    cand_poly = transform_polygon(points, cand_pose)
                    cand_center = cand_pose[:2]
                    if _collides_one_vs_all(state, move, cand_center, cand_poly, overlap_mode=overlap_mode):
                        continue
                    state.poses[move] = cand_pose
                    state.centers[move] = cand_center
                    state.polys[move] = cand_poly
                    state.bboxes[move] = polygon_bbox(cand_poly)
                    moved = True
                    break
                if moved:
                    break

            # Escalate step scales deterministically (small -> larger).
            for step_scale in (1.0, 2.0, 4.0, 8.0):
                step = float(step_xy) * float(step_scale)
                if step <= 0.0:
                    continue
                for d in normed_dirs:
                    cand_pose = base_pose.copy()
                    cand_pose[0:2] = cand_pose[0:2] + d * step
                    cand_poly = transform_polygon(points, cand_pose)
                    cand_center = cand_pose[:2]
                    if _collides_one_vs_all(state, move, cand_center, cand_poly, overlap_mode=overlap_mode):
                        continue

                    state.poses[move] = cand_pose
                    state.centers[move] = cand_center
                    state.polys[move] = cand_poly
                    state.bboxes[move] = polygon_bbox(cand_poly)
                    moved = True
                    break
                if moved:
                    break
            if moved:
                break

            # Fallback: a small random budget helps escape local dead-ends.
            for _try in range(12):
                perp_scale = float(rng.normal(0.0, 0.75))
                d = unit + perp * perp_scale
                d_norm = float(np.linalg.norm(d))
                if d_norm < 1e-12:
                    d = unit
                    d_norm = 1.0
                d = d / d_norm

                step_scale = (0.5, 1.0, 2.0, 4.0, 8.0)[int(rng.integers(0, 5))]
                step = float(step_xy) * float(step_scale)
                noise = rng.normal(0.0, step * 0.10, size=(2,))

                cand_pose = base_pose.copy()
                cand_pose[0:2] = cand_pose[0:2] + d * step + noise
                if step_deg != 0.0:
                    cand_pose[2] = float(math.fmod(cand_pose[2] + rng.normal(0.0, float(step_deg)), 360.0))
                    if cand_pose[2] < 0.0:
                        cand_pose[2] += 360.0

                cand_poly = transform_polygon(points, cand_pose)
                cand_center = cand_pose[:2]
                if _collides_one_vs_all(state, move, cand_center, cand_poly, overlap_mode=overlap_mode):
                    continue

                state.poses[move] = cand_pose
                state.centers[move] = cand_center
                state.polys[move] = cand_poly
                state.bboxes[move] = polygon_bbox(cand_poly)
                moved = True
                break
            if moved:
                break

        # If neither tree can move alone, try a symmetric "pair nudge" that moves
        # both trees by half-steps in opposite directions (often succeeds in very
        # tight local configurations).
        if not moved and n >= 2:
            base_i = state.poses[i].copy()
            base_j = state.poses[j].copy()

            direction = state.centers[i] - state.centers[j]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                ang = float(rng.uniform(0.0, 2.0 * math.pi))
                direction = np.array([math.cos(ang), math.sin(ang)], dtype=float)
                norm = 1.0
            unit = direction / norm
            perp = np.array([-unit[1], unit[0]], dtype=float)

            dirs: list[np.ndarray] = [unit, perp, -perp, unit + perp, unit - perp]
            normed_dirs: list[np.ndarray] = []
            for d in dirs:
                d_norm = float(np.linalg.norm(d))
                if d_norm < 1e-12:
                    continue
                normed_dirs.append(d / d_norm)

            exclude = np.zeros((n,), dtype=bool)
            exclude[i] = True
            exclude[j] = True

            for step_scale in (1.0, 2.0, 4.0, 8.0):
                step = float(step_xy) * float(step_scale)
                if step <= 0.0:
                    continue
                for d in normed_dirs:
                    cand_i = base_i.copy()
                    cand_j = base_j.copy()
                    cand_i[0:2] = cand_i[0:2] + d * step
                    cand_j[0:2] = cand_j[0:2] - d * step

                    poly_i = transform_polygon(points, cand_i)
                    poly_j = transform_polygon(points, cand_j)
                    center_i = cand_i[:2]
                    center_j = cand_j[:2]

                    if intersects(poly_i, poly_j):
                        continue
                    if _collides_candidate_excluding(state, center_i, poly_i, exclude=exclude, overlap_mode=overlap_mode):
                        continue
                    if _collides_candidate_excluding(state, center_j, poly_j, exclude=exclude, overlap_mode=overlap_mode):
                        continue

                    state.poses[i] = cand_i
                    state.centers[i] = center_i
                    state.polys[i] = poly_i
                    state.bboxes[i] = polygon_bbox(poly_i)

                    state.poses[j] = cand_j
                    state.centers[j] = center_j
                    state.polys[j] = poly_j
                    state.bboxes[j] = polygon_bbox(poly_j)

                    moved = True
                    break
                if moved:
                    break

        # Last resort for non-strict modes: allow touching with *other* trees while
        # breaking the current pair, as long as we stay strict-feasible (no area
        # overlaps). This helps unlock highly constrained local configurations.
        if not moved and overlap_mode == "conservative":
            for move in move_order:
                other = j if move == i else i
                base_pose = state.poses[move].copy()

                direction = state.centers[move] - state.centers[other]
                norm = float(np.linalg.norm(direction))
                if norm < 1e-12:
                    ang = float(rng.uniform(0.0, 2.0 * math.pi))
                    direction = np.array([math.cos(ang), math.sin(ang)], dtype=float)
                    norm = 1.0
                unit = direction / norm
                perp = np.array([-unit[1], unit[0]], dtype=float)

                dirs: list[np.ndarray] = [unit, perp, -perp, unit + perp, unit - perp]
                normed_dirs = []
                for d in dirs:
                    d_norm = float(np.linalg.norm(d))
                    if d_norm < 1e-12:
                        continue
                    normed_dirs.append(d / d_norm)

                rot_deltas: tuple[float, ...] = (0.0,)
                if step_deg != 0.0:
                    rot = float(step_deg)
                    rot_deltas = (0.0, rot, -rot)

                    # Rotation-only first.
                    for ddeg in (rot, -rot):
                        cand_pose = base_pose.copy()
                        cand_pose[2] = float(math.fmod(cand_pose[2] + ddeg, 360.0))
                        if cand_pose[2] < 0.0:
                            cand_pose[2] += 360.0
                        cand_poly = transform_polygon(points, cand_pose)
                        cand_center = cand_pose[:2]
                        if _collides_one_vs_all(state, move, cand_center, cand_poly, overlap_mode="strict"):
                            continue
                        if intersects(cand_poly, state.polys[other]):
                            continue
                        state.poses[move] = cand_pose
                        state.centers[move] = cand_center
                        state.polys[move] = cand_poly
                        state.bboxes[move] = polygon_bbox(cand_poly)
                        moved = True
                        break
                    if moved:
                        break

                for step_scale in (1.0, 2.0, 4.0, 8.0):
                    step = float(step_xy) * float(step_scale)
                    if step <= 0.0:
                        continue
                    for d in normed_dirs:
                        for ddeg in rot_deltas:
                            cand_pose = base_pose.copy()
                            cand_pose[0:2] = cand_pose[0:2] + d * step
                            if ddeg != 0.0:
                                cand_pose[2] = float(math.fmod(cand_pose[2] + ddeg, 360.0))
                                if cand_pose[2] < 0.0:
                                    cand_pose[2] += 360.0
                            cand_poly = transform_polygon(points, cand_pose)
                            cand_center = cand_pose[:2]
                            if _collides_one_vs_all(state, move, cand_center, cand_poly, overlap_mode="strict"):
                                continue
                            if intersects(cand_poly, state.polys[other]):
                                continue

                            state.poses[move] = cand_pose
                            state.centers[move] = cand_center
                            state.polys[move] = cand_poly
                            state.bboxes[move] = polygon_bbox(cand_poly)
                            moved = True
                            break
                        if moved:
                            break
                    if moved:
                        break
                if moved:
                    break

        if not moved:
            continue

    return None


def genetic_optimize(
    points: np.ndarray,
    seeds: Sequence[np.ndarray],
    *,
    seed: int = 1,
    pop_size: int = 24,
    generations: int = 20,
    elite_frac: float = 0.25,
    crossover_prob: float = 0.5,
    mutation_sigma_xy: float = 0.01,
    mutation_sigma_deg: float = 2.0,
    directed_mut_prob: float = 0.5,
    directed_step_xy: float = 0.02,
    directed_k: int = 8,
    repair_iters: int = 200,
    hill_climb_passes: int = 0,
    hill_climb_step_xy: float = 0.01,
    hill_climb_step_deg: float = 2.0,
    overlap_mode: OverlapMode = "strict",
    max_child_attempts: int = 50,
) -> np.ndarray:
    """Simple GA for 1 instance: selection + (optional) crossover + mutations (+ optional hill-climb).

    Keeps solutions feasible by repairing (or retrying) colliding children.

    Args:
        points: Local polygon vertices `(V, 2)`.
        seeds: Sequence of initial candidate poses arrays `(N, 3)`.
        seed: RNG seed.
        pop_size: Population size.
        generations: Number of generations.
        elite_frac: Fraction of elites preserved each generation.
        crossover_prob: Probability of crossover vs cloning.
        mutation_sigma_xy: Stddev for xy mutation.
        mutation_sigma_deg: Stddev for rotation mutation.
        directed_mut_prob: Probability of a boundary-directed mutation vs random.
        directed_step_xy: Step size for directed drift.
        directed_k: Neighborhood size for directed selection.
        repair_iters: Repair iteration budget per child.
        hill_climb_passes: Optional hill-climb passes applied to children.
        hill_climb_step_xy: Hill-climb xy step size.
        hill_climb_step_deg: Hill-climb rotation step size.
        max_child_attempts: Retry budget when generating a feasible child.

    Returns:
        Best found poses shifted to the origin.
    """

    if not seeds:
        raise ValueError("empty seeds")
    rng = np.random.default_rng(int(seed))

    seeds_arr = [np.array(p, dtype=float, copy=True) for p in seeds]
    n = int(seeds_arr[0].shape[0])
    for p in seeds_arr:
        if p.shape != (n, 3):
            raise ValueError("all seeds must have the same shape (n,3)")

    def _eval(p: np.ndarray) -> float:
        state = _build_state(points, p)
        if _state_has_overlaps(state, overlap_mode=overlap_mode):
            return float("inf")
        return state.score

    def _tournament(pop: list[np.ndarray], scores: np.ndarray, k: int = 3) -> np.ndarray:
        idxs = rng.integers(0, len(pop), size=(k,))
        best = idxs[0]
        for ii in idxs[1:]:
            if scores[ii] < scores[best]:
                best = ii
        return pop[int(best)]

    def _mutate_directed(p: np.ndarray) -> np.ndarray:
        child = np.array(p, dtype=float, copy=True)
        idx, axis, center = _pick_boundary_tree_from_centers(child[:, :2], rng=rng, k=directed_k)

        drift = np.zeros((2,), dtype=float)
        diff = center - child[idx, :2]
        if axis == 0:
            drift[0] = (
                math.copysign(directed_step_xy, float(diff[0]))
                if abs(float(diff[0])) > 1e-12
                else float(directed_step_xy)
            )
        else:
            drift[1] = (
                math.copysign(directed_step_xy, float(diff[1]))
                if abs(float(diff[1])) > 1e-12
                else float(directed_step_xy)
            )

        noise = rng.normal(0.0, mutation_sigma_xy, size=(2,))
        child[idx, 0:2] = child[idx, 0:2] + drift + noise
        child[idx, 2] = float(math.fmod(child[idx, 2] + rng.normal(0.0, mutation_sigma_deg), 360.0))
        if child[idx, 2] < 0.0:
            child[idx, 2] += 360.0
        return child

    def _mutate_random(p: np.ndarray) -> np.ndarray:
        child = np.array(p, dtype=float, copy=True)
        idx = int(rng.integers(0, n))
        child[idx, 0:2] = child[idx, 0:2] + rng.normal(0.0, mutation_sigma_xy, size=(2,))
        child[idx, 2] = float(math.fmod(child[idx, 2] + rng.normal(0.0, mutation_sigma_deg), 360.0))
        if child[idx, 2] < 0.0:
            child[idx, 2] += 360.0
        return child

    # --- Initialize population
    population: list[np.ndarray] = []
    for p in seeds_arr:
        population.append(shift_poses_to_origin(points, p))
        if len(population) >= pop_size:
            break

    base = population[0]
    while len(population) < pop_size:
        cand = _mutate_directed(base) if rng.random() < directed_mut_prob else _mutate_random(base)
        repaired = _repair_overlaps(
            points,
            cand,
            rng=rng,
            max_iters=repair_iters,
            step_xy=mutation_sigma_xy,
            overlap_mode=overlap_mode,
        )
        if repaired is None:
            continue
        population.append(repaired)

    scores = np.array([_eval(p) for p in population], dtype=float)

    elite_n = max(1, int(round(pop_size * float(elite_frac))))
    best_pose = population[int(np.argmin(scores))]
    best_score = float(np.min(scores))

    # --- Evolve
    for _gen in range(generations):
        order = np.argsort(scores, kind="mergesort")
        population = [population[int(i)] for i in order]
        scores = scores[order]

        next_pop: list[np.ndarray] = [population[i].copy() for i in range(elite_n)]

        while len(next_pop) < pop_size:
            parent_a = _tournament(population, scores)
            parent_b = _tournament(population, scores)

            if rng.random() < crossover_prob:
                mask = rng.random(n) < 0.5
                child = np.where(mask[:, None], parent_a, parent_b).astype(float, copy=True)
            else:
                child = parent_a.copy()

            for _attempt in range(max_child_attempts):
                cand = _mutate_directed(child) if rng.random() < directed_mut_prob else _mutate_random(child)
                repaired = _repair_overlaps(
                    points,
                    cand,
                    rng=rng,
                    max_iters=repair_iters,
                    step_xy=mutation_sigma_xy,
                    overlap_mode=overlap_mode,
                )
                if repaired is None:
                    continue
                cand = repaired
                if hill_climb_passes > 0:
                    cand = hill_climb(
                        points,
                        cand,
                        step_xy=hill_climb_step_xy,
                        step_deg=hill_climb_step_deg,
                        max_passes=hill_climb_passes,
                        overlap_mode=overlap_mode,
                    )
                child = cand
                break

            next_pop.append(child)

        population = next_pop
        scores = np.array([_eval(p) for p in population], dtype=float)
        gen_best_idx = int(np.argmin(scores))
        gen_best = float(scores[gen_best_idx])
        if gen_best < best_score:
            best_score = gen_best
            best_pose = population[gen_best_idx]

    if hill_climb_passes > 0 and not math.isinf(best_score):
        best_pose = hill_climb(
            points,
            best_pose,
            step_xy=hill_climb_step_xy,
            step_deg=hill_climb_step_deg,
            max_passes=hill_climb_passes,
            overlap_mode=overlap_mode,
        )
    return shift_poses_to_origin(points, best_pose)


def _collides_candidate(
    state: _PackingState,
    cand_center: np.ndarray,
    cand_poly: np.ndarray,
    *,
    overlap_mode: OverlapMode,
) -> bool:
    intersects = _intersects_for_mode(overlap_mode)
    cand_bbox = polygon_bbox(cand_poly)
    clearance = float(KAGGLE_CLEARANCE) if overlap_mode == "kaggle" else 0.0
    thr2 = state.thr2
    if clearance > 0.0:
        thr = math.sqrt(float(state.thr2))
        thr2 = (thr + 2.0 * clearance) ** 2
    n = state.centers.shape[0]
    for j in range(n):
        dx = float(cand_center[0] - state.centers[j, 0])
        dy = float(cand_center[1] - state.centers[j, 1])
        if dx * dx + dy * dy > thr2:
            continue
        if not _aabb_overlaps(cand_bbox, state.bboxes[j], eps=float(EPS) + clearance):
            continue
        if intersects(cand_poly, state.polys[j]):
            return True
    return False


def _collides_candidate_excluding(
    state: _PackingState,
    cand_center: np.ndarray,
    cand_poly: np.ndarray,
    exclude: np.ndarray,
    *,
    overlap_mode: OverlapMode,
) -> bool:
    intersects = _intersects_for_mode(overlap_mode)
    cand_bbox = polygon_bbox(cand_poly)
    clearance = float(KAGGLE_CLEARANCE) if overlap_mode == "kaggle" else 0.0
    thr2 = state.thr2
    if clearance > 0.0:
        thr = math.sqrt(float(state.thr2))
        thr2 = (thr + 2.0 * clearance) ** 2
    n = state.centers.shape[0]
    for j in range(n):
        if bool(exclude[j]):
            continue
        dx = float(cand_center[0] - state.centers[j, 0])
        dy = float(cand_center[1] - state.centers[j, 1])
        if dx * dx + dy * dy > thr2:
            continue
        if not _aabb_overlaps(cand_bbox, state.bboxes[j], eps=float(EPS) + clearance):
            continue
        if intersects(cand_poly, state.polys[j]):
            return True
    return False


def _rotate_about_xy(xy: np.ndarray, *, center: np.ndarray, delta_deg: float) -> np.ndarray:
    rad = math.radians(float(delta_deg))
    c = math.cos(rad)
    s = math.sin(rad)
    v = xy - center[None, :]
    x = v[:, 0]
    y = v[:, 1]
    xr = x * c - y * s
    yr = x * s + y * c
    return center[None, :] + np.stack([xr, yr], axis=1)


def _sample_centers(
    *,
    rng: np.random.Generator,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    pad: float,
    n: int,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    xs = rng.uniform(min_x - pad, max_x + pad, size=(n,))
    ys = rng.uniform(min_y - pad, max_y + pad, size=(n,))

    # Bias half the samples toward the current AABB boundary.
    half = n // 2
    for i in range(half):
        if rng.random() < 0.5:
            # Pin x near min/max.
            xs[i] = (min_x - pad) if rng.random() < 0.5 else (max_x + pad)
            ys[i] = rng.uniform(min_y - pad, max_y + pad)
        else:
            # Pin y near min/max.
            ys[i] = (min_y - pad) if rng.random() < 0.5 else (max_y + pad)
            xs[i] = rng.uniform(min_x - pad, max_x + pad)

    return np.stack([xs, ys], axis=1)


def _pick_ruin_set(
    state: _PackingState,
    *,
    rng: np.random.Generator,
    k: int,
    mode: str,
) -> list[int]:
    n = state.poses.shape[0]
    if k <= 0 or n <= 1:
        return []
    k = min(k, n - 1)  # keep at least one tree fixed so we can rebuild reliably

    chosen: list[int] = []
    chosen_set: set[int] = set()

    def _add(idx: int) -> None:
        if idx in chosen_set:
            return
        chosen.append(idx)
        chosen_set.add(idx)

    if mode == "random":
        idxs = rng.choice(n, size=(k,), replace=False)
        return [int(i) for i in idxs]

    if mode == "cluster":
        anchor = int(rng.integers(0, n))
        centers = state.centers
        d2 = np.sum((centers - centers[anchor][None, :]) ** 2, axis=1)
        order = np.argsort(d2, kind="mergesort")
        for idx in order[:k]:
            _add(int(idx))
        return chosen

    # boundary / mixed
    boundary_n = k if mode == "boundary" else max(1, k // 2)
    for _ in range(boundary_n):
        idx, _axis, _center = _pick_boundary_tree(state.bboxes, rng=rng, k=8)
        _add(int(idx))

    while len(chosen) < k:
        _add(int(rng.integers(0, n)))
    return chosen


def large_neighborhood_search(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    seed: int = 1,
    passes: int = 10,
    destroy_k: int = 8,
    destroy_mode: str = "mixed",
    tabu_tenure: int = 0,
    candidates: int = 64,
    angle_samples: int = 8,
    pad_scale: float = 2.0,
    group_moves: int = 0,
    group_size: int = 3,
    group_trans_sigma: float = 0.05,
    group_rot_sigma: float = 20.0,
    t_start: float = 0.0,
    t_end: float = 0.0,
    overlap_mode: OverlapMode = "strict",
) -> np.ndarray:
    """Large Neighborhood Search / (simplified) ALNS-style loop.

    Operators:
      - Ruin & recreate: remove `destroy_k` trees (boundary/random/cluster) and reinsert
        greedily by sampling candidate centers/angles.
      - Group rotation: rotate a small clustered group around its centroid (+ translation).

    Acceptance:
      - Always accept improvements.
      - If `t_start > 0` and `t_end > 0`, accept worse moves with SA probability.

    Notes:
      - `destroy_mode="alns"` adaptively samples among {"boundary","random","cluster"} based on
        recent successful moves (simple reaction-factor update).
      - `tabu_tenure>0` keeps a tabu list of recent ruin-sets (tree-index tuples) to reduce cycles.

    Args:
        points: Local polygon vertices `(V, 2)`.
        poses: Initial poses `(N, 3)`.
        seed: RNG seed.
        passes: Number of destroy/recreate passes.
        destroy_k: Number of trees to remove each pass.
        destroy_mode: Ruin-set sampling strategy.
        tabu_tenure: If >0, keep last `tabu_tenure` ruin-sets as tabu.
        candidates: Candidate center samples used during reinsert.
        angle_samples: Candidate angle samples per reinsert.
        pad_scale: Padding factor applied to sampling bounds.
        group_moves: Number of optional group-rotation moves per pass.
        group_size: Group size for group moves.
        group_trans_sigma: Translation sigma for group moves.
        group_rot_sigma: Rotation sigma for group moves.
        t_start: Optional SA temperature start (0 disables worse acceptance).
        t_end: Optional SA temperature end.

    Returns:
        Best found poses shifted to the origin.
    """

    poses = np.array(poses, dtype=float, copy=True)
    n = int(poses.shape[0])
    if n <= 1 or passes <= 0:
        return shift_poses_to_origin(points, poses)
    if destroy_k <= 0:
        destroy_k = 1

    destroy_mode = str(destroy_mode)
    if destroy_mode not in {"mixed", "boundary", "random", "cluster", "alns"}:
        raise ValueError("destroy_mode must be one of: mixed, boundary, random, cluster, alns")

    tabu_tenure = int(tabu_tenure)
    if tabu_tenure < 0:
        raise ValueError("tabu_tenure must be >= 0")
    tabu: deque[tuple[int, ...]] | None = deque(maxlen=tabu_tenure) if tabu_tenure > 0 else None

    alns_modes: tuple[str, ...] = ("boundary", "random", "cluster")
    alns_weights = np.ones((len(alns_modes),), dtype=float)
    alns_scores = np.zeros((len(alns_modes),), dtype=float)
    alns_counts = np.zeros((len(alns_modes),), dtype=np.int32)
    alns_reaction = 0.2
    alns_update_every = 10

    rng = np.random.default_rng(int(seed))
    rad = float(polygon_radius(points))
    pad = float(pad_scale) * 2.0 * rad

    best = poses
    best_score = _build_state(points, poses).score
    current = poses
    current_score = best_score

    def _temp(pass_idx: int) -> float:
        if t_start <= 0.0 or t_end <= 0.0 or passes <= 1:
            return 0.0
        frac = float(pass_idx) / float(max(passes - 1, 1))
        return float(t_start) * (float(t_end) / float(t_start)) ** frac

    for pass_idx in range(int(passes)):
        base_state = _build_state(points, current)

        # --- ALNS destroy-mode selection (only affects ruin&recreate operator)
        ruin_mode = destroy_mode
        ruin_mode_idx: int | None = None
        if destroy_mode == "alns":
            probs = alns_weights / float(np.sum(alns_weights))
            ruin_mode_idx = int(rng.choice(len(alns_modes), p=probs))
            ruin_mode = alns_modes[ruin_mode_idx]
            alns_counts[ruin_mode_idx] += 1

        # --- Operator 1: group rotations (cheap neighborhood expansion)
        cand_from_group: np.ndarray | None = None
        cand_group_score: float | None = None
        if group_moves > 0 and group_size >= 2:
            for _ in range(int(group_moves)):
                anchor = int(rng.integers(0, n))
                d2 = np.sum((base_state.centers - base_state.centers[anchor][None, :]) ** 2, axis=1)
                order = np.argsort(d2, kind="mergesort")
                group = [int(i) for i in order[: max(2, min(int(group_size), n))]]
                exclude = np.zeros((n,), dtype=bool)
                exclude[group] = True

                dxy = rng.normal(0.0, float(group_trans_sigma), size=(2,))
                ddeg = float(rng.normal(0.0, float(group_rot_sigma)))

                cand = np.array(current, dtype=float, copy=True)
                xy = cand[group, 0:2]
                center = np.mean(xy, axis=0)
                cand_xy = _rotate_about_xy(xy, center=center, delta_deg=ddeg) + dxy[None, :]
                cand[group, 0:2] = cand_xy
                cand[group, 2] = np.mod(cand[group, 2] + ddeg, 360.0)

                # Collision check only vs non-group indices (exclude old group polys).
                ok = True
                cand_bboxes = base_state.bboxes.copy()
                for idx in group:
                    cand_poly = transform_polygon(points, cand[idx])
                    cand_center = cand[idx, 0:2]
                    if _collides_candidate_excluding(
                        base_state,
                        cand_center,
                        cand_poly,
                        exclude=exclude,
                        overlap_mode=overlap_mode,
                    ):
                        ok = False
                        break
                    cand_bboxes[idx] = polygon_bbox(cand_poly)
                if not ok:
                    continue
                cand_score = _packing_score_from_bboxes(cand_bboxes)
                if cand_group_score is None or cand_score < float(cand_group_score):
                    cand_from_group = cand
                    cand_group_score = float(cand_score)

        # --- Operator 2: ruin & recreate (large neighborhood)
        removed_key: tuple[int, ...] | None = None
        if tabu is None:
            removed = _pick_ruin_set(base_state, rng=rng, k=int(destroy_k), mode=ruin_mode)
            removed_key = tuple(sorted(removed)) if removed else None
        else:
            removed = []
            for _ in range(10):
                cand = _pick_ruin_set(base_state, rng=rng, k=int(destroy_k), mode=ruin_mode)
                key = tuple(sorted(cand))
                if key not in tabu:
                    removed = cand
                    removed_key = key
                    break
            if not removed:
                removed = cand
                removed_key = tuple(sorted(removed))

        cand_from_ruin: np.ndarray | None = None
        cand_ruin_score: float | None = None
        if removed:
            removed_set = set(removed)
            keep = [i for i in range(n) if i not in removed_set]
            keep_poses = current[keep]
            keep_state = _build_state(points, keep_poses)

            # Greedy reinsertion in random order.
            insert_order = list(removed)
            rng.shuffle(insert_order)

            cand = np.array(current, dtype=float, copy=True)
            success = True
            for idx in insert_order:
                # Bounding box of current kept packing.
                min_x = float(np.min(keep_state.bboxes[:, 0]))
                min_y = float(np.min(keep_state.bboxes[:, 1]))
                max_x = float(np.max(keep_state.bboxes[:, 2]))
                max_y = float(np.max(keep_state.bboxes[:, 3]))

                # Always include the original pose as a candidate (may fail if earlier inserts moved into it).
                orig_pose = np.array(current[idx], dtype=float, copy=False)
                centers = _sample_centers(
                    rng=rng, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, pad=pad, n=max(0, int(candidates) - 1)
                )
                centers = np.vstack([orig_pose[0:2][None, :], centers])

                n_ang = max(1, int(angle_samples))
                angles = rng.uniform(0.0, 360.0, size=(n_ang - 1,))
                angles = np.concatenate([np.array([float(orig_pose[2])], dtype=float), angles], axis=0)

                best_pose: np.ndarray | None = None
                best_poly: np.ndarray | None = None
                best_bbox: np.ndarray | None = None
                best_score_local = float("inf")

                for cxy in centers:
                    for ang in angles:
                        cand_pose = np.array([float(cxy[0]), float(cxy[1]), float(ang)], dtype=float)
                        cand_poly = transform_polygon(points, cand_pose)
                        cand_center = cand_pose[0:2]
                        if _collides_candidate(keep_state, cand_center, cand_poly, overlap_mode=overlap_mode):
                            continue
                        cand_bbox = polygon_bbox(cand_poly)
                        new_min_x = min(min_x, float(cand_bbox[0]))
                        new_min_y = min(min_y, float(cand_bbox[1]))
                        new_max_x = max(max_x, float(cand_bbox[2]))
                        new_max_y = max(max_y, float(cand_bbox[3]))
                        s = max(new_max_x - new_min_x, new_max_y - new_min_y)
                        if s < best_score_local:
                            best_score_local = float(s)
                            best_pose = cand_pose
                            best_poly = cand_poly
                            best_bbox = cand_bbox

                if best_pose is None or best_poly is None or best_bbox is None:
                    success = False
                    break

                # Commit insertion into keep_state (append).
                keep_state.poses = np.vstack([keep_state.poses, best_pose[None, :]])
                keep_state.centers = np.vstack([keep_state.centers, best_pose[None, 0:2]])
                keep_state.polys.append(best_poly)
                keep_state.bboxes = np.vstack([keep_state.bboxes, best_bbox[None, :]])
                keep_state.score = float(best_score_local)
                cand[idx] = best_pose

            if success:
                cand_from_ruin = cand
                cand_ruin_score = float(keep_state.score)

        # --- Pick best candidate among operators
        best_cand = None
        best_cand_score = None
        if cand_from_group is not None:
            best_cand = cand_from_group
            best_cand_score = float(cand_group_score) if cand_group_score is not None else None
        if cand_from_ruin is not None:
            if best_cand_score is None or (
                cand_ruin_score is not None and float(cand_ruin_score) < float(best_cand_score)
            ):
                best_cand = cand_from_ruin
                best_cand_score = float(cand_ruin_score) if cand_ruin_score is not None else None

        if best_cand is None or best_cand_score is None:
            continue

        # --- Acceptance (greedy or SA)
        delta = float(best_cand_score) - float(current_score)
        accept = delta < 0.0
        temp = _temp(pass_idx)
        if not accept and temp > 0.0:
            accept = rng.random() < math.exp(-delta / temp)

        best_is_ruin = best_cand is cand_from_ruin
        if accept:
            current = best_cand
            current_score = float(best_cand_score)
            if current_score < best_score:
                best = current
                best_score = current_score

            if tabu is not None and best_is_ruin and removed_key is not None:
                tabu.append(removed_key)

        # --- ALNS weight update (periodic reaction-factor update)
        if destroy_mode == "alns" and ruin_mode_idx is not None:
            reward = 0.0
            if accept and best_is_ruin:
                reward = 5.0 if delta < 0.0 else 1.0
            alns_scores[ruin_mode_idx] += reward

            if (pass_idx + 1) % alns_update_every == 0:
                for i in range(len(alns_modes)):
                    avg = float(alns_scores[i]) / float(max(int(alns_counts[i]), 1))
                    avg = max(avg, 1e-3)
                    alns_weights[i] = (1.0 - alns_reaction) * float(alns_weights[i]) + alns_reaction * avg
                alns_weights = np.clip(alns_weights, 1e-3, None)
                alns_scores[:] = 0.0
                alns_counts[:] = 0

    return shift_poses_to_origin(points, best)
