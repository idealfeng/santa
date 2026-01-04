# Santa 2025 - Christmas Tree Packing Challenge (local solver)
# Generates a valid `submission.csv` using public baselines + local search.
#
# Notes
# - The competition provides only `sample_submission.csv`; tree geometry is fixed and defined in the public metric.
# - This script matches the metric's collision logic (shapely + touches allowance) and bounding-square scoring.
#
# Dependencies: pandas, shapely

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from shapely import affinity
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: shapely\n"
        "Install with: pip install shapely\n"
        f"Original import error: {e}"
    )


XY_MIN, XY_MAX = -100.0, 100.0

# Match public metric (santa-2025-metric.ipynb)
getcontext().prec = 25
SCALE_FACTOR = Decimal("1e18")

# Fixed tree bounds (unscaled) for common angles used in grid layouts.
# These are exact for 0/180 since the widest point is base_w/2=0.35 at y=0,
# and the vertical extremes are trunk_bottom=-0.2 and tip_y=0.8.
TREE_BOUNDS: Dict[int, Tuple[float, float, float, float]] = {
    0: (-0.35, -0.2, 0.35, 0.8),
    180: (-0.35, -0.8, 0.35, 0.2),
}


def _bounds_for_fixed_angles(placements: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for x, y, deg in placements:
        b = TREE_BOUNDS.get(int(round(deg)) % 360)
        if b is None:
            poly = ChristmasTree(x, y, deg).polygon
            bx0, by0, bx1, by1 = (v / float(SCALE_FACTOR) for v in poly.bounds)
        else:
            bx0, by0, bx1, by1 = b
        min_x = min(min_x, x + bx0)
        max_x = max(max_x, x + bx1)
        min_y = min(min_y, y + by0)
        max_y = max(max_y, y + by1)
    return min_x, min_y, max_x, max_y


def side_length_for(placements: List[Tuple[float, float, float]]) -> float:
    min_x, min_y, max_x, max_y = _bounds_for_fixed_angles(placements)
    return max(max_x - min_x, max_y - min_y)


def _lattice_points(
    *,
    dx: float = 0.7,
    y_step: float = 1.0,
    y_odd_offset: float = 0.8,
    odd_x_offset: float = 0.35,
    span_x: int = 30,
    span_y: int = 30,
) -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float, float]] = []
    for j in range(-span_y, span_y + 1):
        y_even = j * y_step
        y_odd = j * y_step + y_odd_offset
        for i in range(-span_x, span_x + 1):
            pts.append((i * dx, y_even, 0.0))
            pts.append((i * dx + odd_x_offset, y_odd, 180.0))
    return pts


def build_submission_lattice_greedy(
    n_max: int,
    *,
    dx: float = 0.7,
    y_step: float = 1.0,
    y_odd_offset: float = 0.8,
    odd_x_offset: float = 0.35,
    span_x: int = 30,
    span_y: int = 30,
) -> pd.DataFrame:
    # Build a nested solution by greedily adding lattice points that minimize the current
    # bounding-square side length. Fast and fully collision-free by construction.
    pts = _lattice_points(
        dx=dx,
        y_step=y_step,
        y_odd_offset=y_odd_offset,
        odd_x_offset=odd_x_offset,
        span_x=span_x,
        span_y=span_y,
    )

    # Precompute each point's bounds contribution.
    contrib = []
    for x, y, deg in pts:
        b = TREE_BOUNDS[int(round(deg)) % 360]
        contrib.append((x + b[0], y + b[1], x + b[2], y + b[3]))

    selected_idx: List[int] = []
    used = [False] * len(pts)
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for n in range(1, n_max + 1):
        best_i = None
        best_side = float("inf")
        best_area = float("inf")
        best_aspect = float("inf")

        for i, (cx0, cy0, cx1, cy1) in enumerate(contrib):
            if used[i]:
                continue
            nx0 = min(min_x, cx0)
            ny0 = min(min_y, cy0)
            nx1 = max(max_x, cx1)
            ny1 = max(max_y, cy1)
            w = nx1 - nx0
            h = ny1 - ny0
            side = w if w >= h else h

            if side < best_side:
                best_side = side
                best_area = w * h
                best_aspect = abs(w - h)
                best_i = i
            elif side == best_side:
                area = w * h
                aspect = abs(w - h)
                if area < best_area or (area == best_area and aspect < best_aspect):
                    best_area = area
                    best_aspect = aspect
                    best_i = i

        assert best_i is not None, "Not enough lattice points; increase span_x/span_y."
        used[best_i] = True
        selected_idx.append(best_i)
        cx0, cy0, cx1, cy1 = contrib[best_i]
        min_x = min(min_x, cx0)
        min_y = min(min_y, cy0)
        max_x = max(max_x, cx1)
        max_y = max(max_y, cy1)

    # Emit groups as first n selected points (nested).
    rows = []
    for n in range(1, n_max + 1):
        for i_t, idx in enumerate(selected_idx[:n]):
            x, y, deg = pts[idx]
            rows.append({"id": f"{n:03d}_{i_t}", "x": _format_s(x), "y": _format_s(y), "deg": _format_s(deg)})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def _clamp_xy(x: Decimal, y: Decimal) -> Tuple[Decimal, Decimal]:
    x_f = float(x)
    y_f = float(y)
    if not (XY_MIN <= x_f <= XY_MAX and XY_MIN <= y_f <= XY_MAX):
        x = Decimal(str(min(XY_MAX, max(XY_MIN, x_f))))
        y = Decimal(str(min(XY_MAX, max(XY_MIN, y_f))))
    return x, y


@dataclass
class ChristmasTree:
    center_x: Decimal | str | float = Decimal("0")
    center_y: Decimal | str | float = Decimal("0")
    angle: Decimal | str | float = Decimal("0")
    polygon: Polygon = None  # set in __post_init__

    def __post_init__(self) -> None:
        self.center_x = Decimal(str(self.center_x))
        self.center_y = Decimal(str(self.center_y))
        self.angle = Decimal(str(self.angle))

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at tip
                (Decimal("0.0") * SCALE_FACTOR, tip_y * SCALE_FACTOR),
                # Right side - Top tier
                (top_w / Decimal("2") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (top_w / Decimal("4") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                # Right side - Middle tier
                (mid_w / Decimal("2") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (mid_w / Decimal("4") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                # Right side - Bottom tier
                (base_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Right trunk
                (trunk_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                (trunk_w / Decimal("2") * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
                # Left trunk
                (-(trunk_w / Decimal("2")) * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
                (-(trunk_w / Decimal("2")) * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Left side - Bottom tier
                (-(base_w / Decimal("2")) * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Left side - Middle tier
                (-(mid_w / Decimal("4")) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (-(mid_w / Decimal("2")) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                # Left side - Top tier
                (-(top_w / Decimal("4")) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (-(top_w / Decimal("2")) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * SCALE_FACTOR),
            yoff=float(self.center_y * SCALE_FACTOR),
        )


def generate_weighted_angle() -> float:
    while True:
        angle = random.uniform(0.0, 2.0 * math.pi)
        if random.uniform(0.0, 1.0) < abs(math.sin(2.0 * angle)):
            return angle


def _has_overlap(candidate_poly: Polygon, tree_index: STRtree, placed_polygons: List[Polygon]) -> bool:
    possible_indices = tree_index.query(candidate_poly)
    for i in possible_indices:
        other = placed_polygons[i]
        if candidate_poly.intersects(other) and not candidate_poly.touches(other):
            return True
    return False


def initialize_trees(
    num_trees: int,
    *,
    existing_trees: Optional[Iterable[ChristmasTree]] = None,
    attempts: int = 10,
    start_radius: Decimal = Decimal("20.0"),
    step_in: Decimal = Decimal("0.5"),
    step_out: Decimal = Decimal("0.05"),
) -> Tuple[List[ChristmasTree], Decimal]:
    if num_trees <= 0:
        return [], Decimal("0")

    placed_trees = list(existing_trees) if existing_trees is not None else []
    num_to_add = num_trees - len(placed_trees)
    if num_to_add < 0:
        placed_trees = placed_trees[:num_trees]
        num_to_add = 0

    if num_to_add > 0:
        unplaced = [ChristmasTree(angle=Decimal(str(random.uniform(0.0, 360.0)))) for _ in range(num_to_add)]
        if not placed_trees:
            placed_trees.append(unplaced.pop(0))  # first at origin

        for tree_to_place in unplaced:
            placed_polygons = [t.polygon for t in placed_trees]
            tree_index = STRtree(placed_polygons)

            best_px = Decimal("0")
            best_py = Decimal("0")
            min_radius = Decimal("Infinity")

            for _ in range(max(1, attempts)):
                a = generate_weighted_angle()
                vx = Decimal(str(math.cos(a)))
                vy = Decimal(str(math.sin(a)))

                radius = start_radius
                collision_found = False

                while radius >= 0:
                    px = radius * vx
                    py = radius * vy
                    px, py = _clamp_xy(px, py)

                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * SCALE_FACTOR),
                        yoff=float(py * SCALE_FACTOR),
                    )
                    if _has_overlap(candidate_poly, tree_index, placed_polygons):
                        collision_found = True
                        break
                    radius -= step_in

                if collision_found:
                    while True:
                        radius += step_out
                        px = radius * vx
                        py = radius * vy
                        px, py = _clamp_xy(px, py)
                        candidate_poly = affinity.translate(
                            tree_to_place.polygon,
                            xoff=float(px * SCALE_FACTOR),
                            yoff=float(py * SCALE_FACTOR),
                        )
                        if not _has_overlap(candidate_poly, tree_index, placed_polygons):
                            break
                else:
                    radius = Decimal("0")
                    px = Decimal("0")
                    py = Decimal("0")

                if radius < min_radius:
                    min_radius = radius
                    best_px = px
                    best_py = py

            tree_to_place.center_x = best_px
            tree_to_place.center_y = best_py
            tree_to_place.polygon = affinity.translate(
                tree_to_place.polygon,
                xoff=float(tree_to_place.center_x * SCALE_FACTOR),
                yoff=float(tree_to_place.center_y * SCALE_FACTOR),
            )
            placed_trees.append(tree_to_place)

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds
    side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    side_length = Decimal(str(side_length_scaled)) / SCALE_FACTOR
    return placed_trees, side_length


def _format_s(v: float) -> str:
    return f"s{v:.6f}"


def build_submission_greedy_prefix(n_max: int, *, seed: int, attempts: int) -> pd.DataFrame:
    random.seed(seed)

    index = [f"{n:03d}_{t}" for n in range(1, n_max + 1) for t in range(n)]

    tree_data: List[Tuple[float, float, float]] = []
    current: List[ChristmasTree] = []
    for n in range(1, n_max + 1):
        current, _side = initialize_trees(n, existing_trees=current, attempts=attempts)
        for tree in current:
            tree_data.append((float(tree.center_x), float(tree.center_y), float(tree.angle)))

    df = pd.DataFrame(index=index, columns=["x", "y", "deg"], data=tree_data).rename_axis("id")
    for col in ["x", "y", "deg"]:
        df[col] = df[col].astype(float).round(6).map(_format_s)
    return df


def _grid_layout(
    n: int,
    *,
    n_even: int,
    n_odd: int,
    dx: float = 0.7,
    y_step: float = 1.0,
    y_odd_offset: float = 0.8,
    odd_x_offset: Optional[float] = None,
) -> List[Tuple[float, float, float]]:
    if n_odd <= 0 or n_even <= 0:
        return []

    if odd_x_offset is None:
        odd_x_offset = dx / 2.0

    placements: List[Tuple[float, float, float]] = []
    rest = n
    r = 0
    while rest > 0:
        m = min(rest, n_even if (r % 2 == 0) else n_odd)
        rest -= m

        angle = 0.0 if (r % 2 == 0) else 180.0
        x_offset = 0.0 if (r % 2 == 0) else odd_x_offset
        if r % 2 == 0:
            y = (r // 2) * y_step
        else:
            y = y_odd_offset + ((r - 1) // 2) * y_step

        for i in range(m):
            placements.append((dx * i + x_offset, y, angle))
        r += 1

    return placements


def _bounds_for_grid(
    n: int,
    *,
    n_even: int,
    n_odd: int,
    dx: float = 0.7,
    y_step: float = 1.0,
    y_odd_offset: float = 0.8,
    odd_x_offset: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    if odd_x_offset is None:
        odd_x_offset = dx / 2.0

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    rest = n
    r = 0
    while rest > 0:
        m = min(rest, n_even if (r % 2 == 0) else n_odd)
        rest -= m

        angle = 0 if (r % 2 == 0) else 180
        x_offset = 0.0 if (r % 2 == 0) else odd_x_offset
        if r % 2 == 0:
            y = (r // 2) * y_step
        else:
            y = y_odd_offset + ((r - 1) // 2) * y_step

        bx0, by0, bx1, by1 = TREE_BOUNDS[angle]
        min_x = min(min_x, x_offset + bx0)
        max_x = max(max_x, x_offset + dx * (m - 1) + bx1)
        min_y = min(min_y, y + by0)
        max_y = max(max_y, y + by1)

        r += 1

    return min_x, min_y, max_x, max_y


def solve_n_grid(n: int, *, span: int = 1) -> List[Tuple[float, float, float]]:
    # Port of public notebook "88.32999 A Well-Aligned Initial Solution".
    best_side2 = float("inf")
    best_pair: Optional[Tuple[int, int]] = None

    for n_even in range(1, n + 1):
        lo = max(1, n_even - span)
        hi = min(n, n_even + span)
        for n_odd in range(lo, hi + 1):
            if n_odd <= 0:
                continue
            min_x, min_y, max_x, max_y = _bounds_for_grid(n, n_even=n_even, n_odd=n_odd)
            side = max(max_x - min_x, max_y - min_y)
            side2 = side * side
            if side2 < best_side2:
                best_side2 = side2
                best_pair = (n_even, n_odd)

    assert best_pair is not None
    return _grid_layout(n, n_even=best_pair[0], n_odd=best_pair[1])


def build_submission_grid(n_max: int, *, span: int = 1) -> pd.DataFrame:
    rows = []
    for n in range(1, n_max + 1):
        placements = solve_n_grid(n, span=span)
        for i, (x, y, deg) in enumerate(placements):
            rows.append({"id": f"{n:03d}_{i}", "x": _format_s(x), "y": _format_s(y), "deg": _format_s(deg)})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def build_submission_grid_prune(n_max: int, *, span: int = 1) -> pd.DataFrame:
    # Build only n_max using grid, then derive n_max-1..1 by deleting one tree at a time
    # that minimizes the bounding-square side length (cheap and often improves total score).
    current = solve_n_grid(n_max, span=span)
    solutions: Dict[int, List[Tuple[float, float, float]]] = {n_max: current}

    for n in range(n_max - 1, 0, -1):
        prev = solutions[n + 1]
        best_i = 0
        best_side = float("inf")
        for i in range(len(prev)):
            cand = prev[:i] + prev[i + 1 :]
            s = side_length_for(cand)
            if s < best_side:
                best_side = s
                best_i = i
        solutions[n] = prev[:best_i] + prev[best_i + 1 :]

    rows = []
    for n in range(1, n_max + 1):
        placements = solutions[n]
        for i, (x, y, deg) in enumerate(placements):
            rows.append({"id": f"{n:03d}_{i}", "x": _format_s(x), "y": _format_s(y), "deg": _format_s(deg)})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def build_submission_grid_propagate(n_max: int, *, span: int = 1) -> pd.DataFrame:
    # Start from best-known grid for each n, then try to improve n-1 using n by deletion.
    solutions: Dict[int, List[Tuple[float, float, float]]] = {}
    sides: Dict[int, float] = {}
    for n in range(1, n_max + 1):
        placements = solve_n_grid(n, span=span)
        solutions[n] = placements
        sides[n] = side_length_for(placements)

    for n in range(n_max, 1, -1):
        src = solutions[n]
        best = solutions[n - 1]
        best_side = sides[n - 1]

        for i in range(len(src)):
            cand = src[:i] + src[i + 1 :]
            s = side_length_for(cand)
            if s < best_side:
                best_side = s
                best = cand

        solutions[n - 1] = best
        sides[n - 1] = best_side

    rows = []
    for n in range(1, n_max + 1):
        placements = solutions[n]
        for i, (x, y, deg) in enumerate(placements):
            rows.append({"id": f"{n:03d}_{i}", "x": _format_s(x), "y": _format_s(y), "deg": _format_s(deg)})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def score_submission(submission: pd.DataFrame) -> float:
    df = submission.copy()
    if "id" in df.columns:
        ids = df["id"].astype(str)
    else:
        ids = df.index.astype(str)
        df = df.reset_index(drop=True)

    data_cols = ["x", "y", "deg"]
    for c in data_cols:
        s = df[c].astype(str)
        if not s.str.startswith("s").all():
            raise ValueError(f"Column {c} contains value(s) without 's' prefix.")
        df[c] = s.str[1:]

    limit = 100.0
    x = df["x"].astype(float)
    y = df["y"].astype(float)
    if (x < -limit).any() or (x > limit).any() or (y < -limit).any() or (y > limit).any():
        raise ValueError("x and/or y values outside the bounds of -100 to 100.")

    groups = ids.str.split("_").str[0]
    df["_group"] = groups.values

    total = Decimal("0.0")
    for group, g in df.groupby("_group", sort=True):
        placed: List[ChristmasTree] = []
        for _, row in g.iterrows():
            placed.append(ChristmasTree(row["x"], row["y"], row["deg"]))

        all_polygons = [t.polygon for t in placed]
        r_tree = STRtree(all_polygons)
        for i, poly in enumerate(all_polygons):
            for j in r_tree.query(poly):
                if j == i:
                    continue
                if poly.intersects(all_polygons[j]) and not poly.touches(all_polygons[j]):
                    raise ValueError(f"Overlapping trees in group {group}")

        bounds = unary_union(all_polygons).bounds
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        group_score = (
            (Decimal(str(side_length_scaled)) ** 2)
            / (SCALE_FACTOR**2)
            / Decimal(str(len(g)))
        )
        total += group_score

    return float(total)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-in",
        default=None,
        help="Read a CSV and write a Kaggle-ready submission (id,x,y,deg) to --out, then exit.",
    )
    parser.add_argument(
        "--score-file",
        default=None,
        help="Score an existing submission CSV (id,x,y,deg with 's' prefixes) and exit.",
    )
    parser.add_argument("--out", default="submission.csv", help="Output CSV path.")
    parser.add_argument(
        "--mode",
        choices=["grid", "lattice_greedy", "grid_propagate", "grid_prune", "greedy_prefix"],
        default="grid",
        help="Submission generator: grid (strong baseline) or greedy_prefix (getting-started style).",
    )
    parser.add_argument(
        "--grid-span",
        type=int,
        default=1,
        help="Grid mode: allow odd-row length to differ by up to this value from even-row length (default 1).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--nmax", type=int, default=200, help="Max trees (default 200).")
    parser.add_argument("--attempts", type=int, default=10, help="Attempts per added tree.")
    parser.add_argument("--score", action="store_true", help="Compute and print the metric score locally.")
    args = parser.parse_args(argv)

    if not (1 <= args.nmax <= 200):
        raise SystemExit("--nmax must be in [1, 200].")

    if args.clean_in is not None:
        src = pd.read_csv(args.clean_in)
        df = src[["id", "x", "y", "deg"]].copy()
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} (rows={len(df)})")
        if args.score:
            print(f"Score: {score_submission(df):.12f}")
        return 0

    if args.score_file is not None:
        df = pd.read_csv(args.score_file)
        df = df[["id", "x", "y", "deg"]].copy()
        print(f"Score: {score_submission(df):.12f}")
        return 0

    out_path = Path(args.out)
    if args.mode == "grid":
        if args.grid_span < 0:
            raise SystemExit("--grid-span must be >= 0.")
        df = build_submission_grid(args.nmax, span=args.grid_span)
    elif args.mode == "lattice_greedy":
        df = build_submission_lattice_greedy(args.nmax)
    elif args.mode == "grid_propagate":
        if args.grid_span < 0:
            raise SystemExit("--grid-span must be >= 0.")
        df = build_submission_grid_propagate(args.nmax, span=args.grid_span)
    elif args.mode == "grid_prune":
        if args.grid_span < 0:
            raise SystemExit("--grid-span must be >= 0.")
        df = build_submission_grid_prune(args.nmax, span=args.grid_span)
    else:
        df = build_submission_greedy_prefix(args.nmax, seed=args.seed, attempts=args.attempts)
    df.to_csv(out_path)
    print(f"Wrote {out_path} (rows={len(df)})")
    if args.score:
        print(f"Score: {score_submission(df):.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
