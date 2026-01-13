"""Local scoring and validation for `submission.csv`.

This module implements:
- CSV parsing (including the `s`-prefixed numeric format)
- packing score computation (`s_n` per puzzle and the official prefix objective)
- robust overlap detection modes (strict / conservative / kaggle)

The scoring code is intentionally NumPy-based to match typical Kaggle
implementations and to keep CLI tools lightweight.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from .constants import EPS
from .geom_np import packing_score, polygon_bbox, polygon_radius, prefix_score, transform_polygon
from .tree_data import TREE_POINTS

try:
    from .fastcollide import polygons_intersect as _polygons_intersect_fast  # type: ignore[attr-defined]
except Exception:
    _polygons_intersect_fast = None


OverlapMode = Literal["strict", "conservative", "kaggle"]

# Semantics:
# - strict: touching is allowed (boundary contact is NOT overlap)
# - conservative: touching counts as overlap (more robust, less dense)
# - kaggle: same as strict (touching allowed); named for parity with the Kaggle evaluator
#
# Historical note: previous versions used a small "clearance" margin for kaggle
# to be extra conservative. The official evaluator allows touching, so the
# margin is disabled by default but kept as a knob for experimentation.
KAGGLE_CLEARANCE: float = 0.0


def _parse_val(value: str) -> float:
    """Parse a competition-formatted scalar (optional `s` prefix) into a float."""
    value = value.strip()
    if value.startswith("s") or value.startswith("S"):
        value = value[1:]
    return float(value)


def load_submission(csv_path: Path, *, nmax: int | None = None) -> dict[int, np.ndarray]:
    """Load a `submission.csv` into a mapping `{n -> poses}`.

    Args:
        csv_path: Path to the CSV file.
        nmax: Optional maximum puzzle id `n` to load.

    Returns:
        A dict mapping puzzle size `n` to a `(n, 3)` float array of `[x, y, deg]`.
    """
    puzzles: dict[int, list[list[float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle_id = int(row["id"].split("_")[0])
            if nmax is not None and puzzle_id > nmax:
                continue
            puzzles[puzzle_id].append(
                [
                    _parse_val(row["x"]),
                    _parse_val(row["y"]),
                    _parse_val(row["deg"]),
                ]
            )
    return {pid: np.array(rows, dtype=float) for pid, rows in puzzles.items()}


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _segments_intersect_strict(p1, p2, p3, p4, eps: float = EPS) -> bool:
    """Proper segment intersection (excludes touching/collinearity).

    Uses robust orientation signs with tolerance: only counts if each segment's endpoints
    are strictly on opposite sides of the other segment's supporting line.
    """
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    s1 = _sign(float(d1), eps)
    s2 = _sign(float(d2), eps)
    s3 = _sign(float(d3), eps)
    s4 = _sign(float(d4), eps)
    return (s1 * s2 < 0) and (s3 * s4 < 0)


def _point_on_segment(p1: np.ndarray, p2: np.ndarray, p: np.ndarray, eps: float = EPS) -> bool:
    # Collinear + within segment AABB (with tolerance).
    if abs(_cross(p1, p2, p)) > eps:
        return False
    min_x = min(float(p1[0]), float(p2[0])) - eps
    max_x = max(float(p1[0]), float(p2[0])) + eps
    min_y = min(float(p1[1]), float(p2[1])) - eps
    max_y = max(float(p1[1]), float(p2[1])) + eps
    return (min_x <= float(p[0]) <= max_x) and (min_y <= float(p[1]) <= max_y)


def _segments_intersect_inclusive(p1, p2, p3, p4, eps: float = EPS) -> bool:
    """Segment intersection treating touching/collinearity as intersection."""
    d1 = _cross(p1, p2, p3)
    d2 = _cross(p1, p2, p4)
    d3 = _cross(p3, p4, p1)
    d4 = _cross(p3, p4, p2)

    s1 = _sign(float(d1), eps)
    s2 = _sign(float(d2), eps)
    s3 = _sign(float(d3), eps)
    s4 = _sign(float(d4), eps)

    if (s1 * s2 < 0) and (s3 * s4 < 0):
        return True

    if s1 == 0 and _point_on_segment(p1, p2, p3, eps):
        return True
    if s2 == 0 and _point_on_segment(p1, p2, p4, eps):
        return True
    if s3 == 0 and _point_on_segment(p3, p4, p1, eps):
        return True
    if s4 == 0 and _point_on_segment(p3, p4, p2, eps):
        return True

    return False


def _point_in_polygon_strict(point: np.ndarray, poly: np.ndarray, eps: float = EPS) -> bool:
    # Treat boundary as outside for the strict predicate.
    n = poly.shape[0]
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        if _point_on_segment(p1, p2, point, eps):
            return False

    x, y = point
    inside = False
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cond1 = (p1[1] > y) != (p2[1] > y)
        if not cond1:
            continue
        x_int = (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1] + eps) + p1[0]
        if x + eps < x_int:
            inside = not inside
    return inside


def _point_in_polygon_inclusive(point: np.ndarray, poly: np.ndarray, eps: float = EPS) -> bool:
    # Treat boundary as inside (robust against "touch" cases).
    n = poly.shape[0]
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        if _point_on_segment(p1, p2, point, eps):
            return True
    return _point_in_polygon_strict(point, poly, eps)


def _polygons_intersect_strict(poly1: np.ndarray, poly2: np.ndarray, eps: float = EPS) -> bool:
    min1 = np.min(poly1, axis=0)
    max1 = np.max(poly1, axis=0)
    min2 = np.min(poly2, axis=0)
    max2 = np.max(poly2, axis=0)
    if not (np.all(max1 >= min2) and np.all(max2 >= min1)):
        return False

    n1 = poly1.shape[0]
    n2 = poly2.shape[0]
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            if _segments_intersect_strict(p1, p2, p3, p4, eps):
                return True

    if _point_in_polygon_strict(poly1[0], poly2, eps):
        return True
    if _point_in_polygon_strict(poly2[0], poly1, eps):
        return True
    return False


def _polygons_intersect_conservative(poly1: np.ndarray, poly2: np.ndarray, eps: float = EPS) -> bool:
    # AABB reject (keep touch within eps as "possible collision").
    min1 = np.min(poly1, axis=0)
    max1 = np.max(poly1, axis=0)
    min2 = np.min(poly2, axis=0)
    max2 = np.max(poly2, axis=0)
    if np.any(max1 < (min2 - eps)) or np.any(max2 < (min1 - eps)):
        return False

    n1 = poly1.shape[0]
    n2 = poly2.shape[0]
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            if _segments_intersect_inclusive(p1, p2, p3, p4, eps):
                return True

    if _point_in_polygon_inclusive(poly1[0], poly2, eps):
        return True
    if _point_in_polygon_inclusive(poly2[0], poly1, eps):
        return True
    return False


def polygons_intersect_strict(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """Fast polygon intersection check (strict).

    This predicate excludes boundary-touching/collinearity and is used as a fast
    path (including an optional C++ accelerator).

    Args:
        poly1: Vertices `(V1, 2)`.
        poly2: Vertices `(V2, 2)`.

    Returns:
        True if the polygons strictly intersect.
    """
    # Prefer the optional C++ extension (when available) in hot paths like lattice
    # spacing search and insertion heuristics.
    #
    # Note: the C++ predicate is intentionally conservative (it may classify some
    # boundary-touch cases as intersections for concave polygons). For "strict"
    # semantics (touching is allowed), we confirm positive hits with the pure
    # NumPy predicate to avoid false positives.
    if _polygons_intersect_fast is not None:
        if not bool(_polygons_intersect_fast(poly1, poly2)):
            return False
        return _polygons_intersect_strict(poly1, poly2, EPS)
    return _polygons_intersect_strict(poly1, poly2, EPS)


def polygons_intersect(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """Conservative intersection check: counts touching as collision (robustness > density)."""
    # For touch-heavy packings, doing a strict pre-check can double the work
    # (strict -> false, then inclusive -> true/false). The inclusive predicate
    # already detects strict intersections, so we directly use it here.
    return _polygons_intersect_conservative(poly1, poly2, EPS)


def _aabb_overlaps(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> bool:
    # Keep "touch" within eps as overlap candidate.
    return not (
        float(a[2]) < float(b[0]) - eps
        or float(b[2]) < float(a[0]) - eps
        or float(a[3]) < float(b[1]) - eps
        or float(b[3]) < float(a[1]) - eps
    )


def _grid_candidate_pairs(centers: np.ndarray, *, cell_size: float) -> set[tuple[int, int]]:
    from collections import defaultdict

    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    inv = 1.0 / float(max(cell_size, 1e-12))
    for idx, (x, y) in enumerate(centers):
        gx = int(math.floor(float(x) * inv))
        gy = int(math.floor(float(y) * inv))
        grid[(gx, gy)].append(int(idx))

    pairs: set[tuple[int, int]] = set()
    for (gx, gy), idxs in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nbr = (gx + dx, gy + dy)
                other = grid.get(nbr)
                if not other:
                    continue
                for i in idxs:
                    for j in other:
                        if j <= i:
                            continue
                        pairs.add((i, j))
    return pairs


def _point_segment_distance_sq(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab[0] * ab[0] + ab[1] * ab[1])
    if denom <= 0.0:
        dx = float(p[0] - a[0])
        dy = float(p[1] - a[1])
        return dx * dx + dy * dy
    ap = p - a
    t = float((ap[0] * ab[0] + ap[1] * ab[1]) / denom)
    if t <= 0.0:
        dx = float(p[0] - a[0])
        dy = float(p[1] - a[1])
        return dx * dx + dy * dy
    if t >= 1.0:
        dx = float(p[0] - b[0])
        dy = float(p[1] - b[1])
        return dx * dx + dy * dy
    proj = a + t * ab
    dx = float(p[0] - proj[0])
    dy = float(p[1] - proj[1])
    return dx * dx + dy * dy


def polygons_min_distance_sq(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Return squared minimum distance between polygon boundaries.

    This can be used to implement a small clearance margin (optional). For
    feasibility checks we only need a conservative bound, so we compute the
    minimum over endpoint-to-segment distances (sufficient in 2D for segments).
    """
    poly1 = np.array(poly1, dtype=float, copy=False)
    poly2 = np.array(poly2, dtype=float, copy=False)
    n1 = int(poly1.shape[0])
    n2 = int(poly2.shape[0])
    if n1 <= 0 or n2 <= 0:
        return float("inf")

    best = float("inf")
    for i in range(n1):
        a1 = poly1[i]
        a2 = poly1[(i + 1) % n1]
        for j in range(n2):
            b1 = poly2[j]
            b2 = poly2[(j + 1) % n2]
            best = min(best, _point_segment_distance_sq(a1, b1, b2))
            best = min(best, _point_segment_distance_sq(a2, b1, b2))
            best = min(best, _point_segment_distance_sq(b1, a1, a2))
            best = min(best, _point_segment_distance_sq(b2, a1, a2))
            if best <= 0.0:
                return 0.0
    return float(best)


def first_overlap_pair(
    points: np.ndarray,
    poses: np.ndarray,
    *,
    eps: float = EPS,
    mode: OverlapMode = "conservative",
) -> tuple[int, int] | None:
    """Return first (i,j) overlap/touch found, or None if feasible.

    Two-stage:
    - Fast: grid neighbors + circle + AABB.
    - Fine: polygon-polygon (strict or conservative).
    """
    poses = np.array(poses, dtype=float, copy=False)
    if poses.shape[0] <= 1:
        return None

    points = np.array(points, dtype=float, copy=False)
    points_check = points
    clearance = float(KAGGLE_CLEARANCE) if mode == "kaggle" else 0.0

    centers = poses[:, :2]
    rad = float(polygon_radius(points_check)) + clearance
    dist_thr = 2.0 * rad + float(eps)
    thr2 = dist_thr * dist_thr

    pairs = _grid_candidate_pairs(centers, cell_size=dist_thr)
    if not pairs:
        return None

    polys = [transform_polygon(points_check, pose) for pose in poses]
    bboxes = [polygon_bbox(p) for p in polys]

    if mode in {"strict", "kaggle"}:
        intersects = polygons_intersect_strict
    elif mode == "conservative":
        # Treat touching as collision for robustness.
        intersects = polygons_intersect
    else:
        raise ValueError(f"Unknown overlap mode: {mode!r}")

    for i, j in sorted(pairs):
        dx = float(centers[i, 0] - centers[j, 0])
        dy = float(centers[i, 1] - centers[j, 1])
        if dx * dx + dy * dy > thr2:
            continue
        if not _aabb_overlaps(bboxes[i], bboxes[j], float(eps) + clearance):
            continue
        if intersects(polys[i], polys[j]):
            return i, j
        if clearance > 0.0:
            if polygons_min_distance_sq(polys[i], polys[j]) <= clearance * clearance:
                return i, j
    return None


def _check_overlaps(points: np.ndarray, poses: np.ndarray, *, mode: OverlapMode) -> bool:
    return first_overlap_pair(points, poses, eps=EPS, mode=mode) is not None


@dataclass
class ScoreResult:
    """Aggregate score information returned by `score_submission`."""

    nmax: int
    score: float
    s_max: float
    overlap_check: bool
    overlap_mode: OverlapMode
    require_complete: bool
    per_n: list[dict]

    def to_json(self) -> dict:
        """Convert the result to a JSON-serializable dict."""
        return {
            "nmax": self.nmax,
            "score": self.score,
            "s_max": self.s_max,
            "overlap_check": self.overlap_check,
            "overlap_mode": self.overlap_mode,
            "require_complete": self.require_complete,
            "per_n": self.per_n,
        }


def score_submission(
    csv_path: Path,
    *,
    nmax: int | None = None,
    check_overlap: bool = True,
    overlap_mode: OverlapMode = "strict",
    require_complete: bool = True,
) -> ScoreResult:
    """Score a `submission.csv` using the local evaluator.

    Args:
        csv_path: Path to `submission.csv`.
        nmax: Optional max puzzle `n` to score.
        check_overlap: If True, validate there is no overlap for each puzzle.
        overlap_mode: Overlap predicate used when `check_overlap=True`.
        require_complete: If True, require every puzzle `1..nmax` to be present.

    Returns:
        A `ScoreResult` with the total score and per-puzzle breakdown.

    Raises:
        ValueError: For missing puzzles, wrong row counts, or detected overlaps.
    """
    points = np.array(TREE_POINTS, dtype=float)
    puzzles = load_submission(csv_path, nmax=nmax)
    if not puzzles:
        return ScoreResult(0, 0.0, 0.0, check_overlap, overlap_mode, require_complete, [])

    max_n = max(puzzles)
    if nmax is None:
        nmax = max_n
    if require_complete:
        missing = [n for n in range(1, nmax + 1) if n not in puzzles]
        if missing:
            raise ValueError(f"Missing puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    total = 0.0
    s_max = 0.0
    per_n: list[dict] = []
    for n in range(1, nmax + 1):
        poses = puzzles.get(n)
        if poses is None:
            continue
        if poses.shape[0] != n:
            raise ValueError(f"Puzzle {n} expected {n} trees, got {poses.shape[0]}")
        if check_overlap and _check_overlaps(points, poses, mode=overlap_mode):
            raise ValueError(f"Overlap detected in puzzle {n}")
        s = packing_score(points, poses)
        s_max = max(s_max, s)
        contrib = (s * s) / n
        total += contrib
        per_n.append({"puzzle": n, "s": s, "contrib": contrib})

    return ScoreResult(nmax, total, s_max, check_overlap, overlap_mode, require_complete, per_n)


def score_prefix(s_values: Iterable[float]) -> float:
    """Compute the official prefix objective from a sequence of `s_n` values."""
    return prefix_score(s_values)
