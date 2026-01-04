# train.py  (single-file solver)
# Santa 2025 - Christmas Tree Packing Challenge
# Dependencies: numpy, pandas

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
XY_MIN, XY_MAX = -100.0, 100.0
EPS_OVERLAP = 1e-10  # treat almost-touching as overlap to avoid evaluation precision issues

# SA / ALNS knobs (good default starting point; tune later)
DEFAULT_BASE_STEPS = 27000
DEFAULT_ATTEMPTS_PER_N = 2

# Seeds to try (outer loop)
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class TreeParam:
    # "tree" is modeled as an isosceles triangle:
    # base corners (-r, 0), (r, 0), apex (0, h)
    r: float
    h: float


@dataclass
class Placement:
    x: float
    y: float
    deg: float  # rotation degrees around local origin (base center)


# -----------------------------
# Geometry helpers
# -----------------------------
def clip_xy(x: float, y: float) -> Tuple[float, float]:
    return (
        float(min(XY_MAX, max(XY_MIN, x))),
        float(min(XY_MAX, max(XY_MIN, y))),
    )


def rot2(deg: float) -> Tuple[float, float, float, float]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return c, -s, s, c


def triangle_vertices(param: TreeParam, place: Placement) -> np.ndarray:
    # local triangle points
    # base centered at origin, apex above
    p0 = (-param.r, 0.0)
    p1 = (param.r, 0.0)
    p2 = (0.0, param.h)

    c00, c01, c10, c11 = rot2(place.deg)

    def tf(p):
        x = c00 * p[0] + c01 * p[1] + place.x
        y = c10 * p[0] + c11 * p[1] + place.y
        return (x, y)

    v0 = tf(p0)
    v1 = tf(p1)
    v2 = tf(p2)
    return np.array([v0, v1, v2], dtype=np.float64)


def compute_vertices(params: List[TreeParam], places: List[Placement]) -> np.ndarray:
    n = len(params)
    verts = np.zeros((n, 3, 2), dtype=np.float64)
    for i in range(n):
        verts[i] = triangle_vertices(params[i], places[i])
    return verts


def bbox_from_verts(verts: np.ndarray) -> Tuple[float, float, float, float]:
    xs = verts[:, :, 0]
    ys = verts[:, :, 1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def bbox_center(verts: np.ndarray) -> Tuple[float, float]:
    minx, maxx, miny, maxy = bbox_from_verts(verts)
    return float((minx + maxx) * 0.5), float((miny + maxy) * 0.5)


def bbox_side(verts: np.ndarray) -> float:
    minx, maxx, miny, maxy = bbox_from_verts(verts)
    return float(max(maxx - minx, maxy - miny))


def score_term(side: float, n: int) -> float:
    return (side * side) / float(n)


def project_on_axis(poly: np.ndarray, ax: np.ndarray) -> Tuple[float, float]:
    # ax must be normalized or not; projection is linear anyway
    dots = poly @ ax
    return float(dots.min()), float(dots.max())


def polys_overlap_sat(a: np.ndarray, b: np.ndarray, eps: float = EPS_OVERLAP) -> bool:
    # SAT for convex polygons (triangles)
    # edges from both polygons => normals as separating axes
    for poly in (a, b):
        for k in range(3):
            p = poly[k]
            q = poly[(k + 1) % 3]
            e = q - p
            # perpendicular axis
            ax = np.array([-e[1], e[0]], dtype=np.float64)
            # if degenerate, skip
            nrm = ax[0] * ax[0] + ax[1] * ax[1]
            if nrm <= 1e-18:
                continue
            ax /= math.sqrt(nrm)

            amin, amax = project_on_axis(a, ax)
            bmin, bmax = project_on_axis(b, ax)
            # separated?
            if amax < bmin + eps or bmax < amin + eps:
                return False
    return True


def any_overlap_idx(i: int, verts: np.ndarray) -> bool:
    vi = verts[i]
    for j in range(verts.shape[0]):
        if j == i:
            continue
        if polys_overlap_sat(vi, verts[j]):
            return True
    return False


def any_overlap_all(verts: np.ndarray) -> bool:
    n = verts.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if polys_overlap_sat(verts[i], verts[j]):
                return True
    return False


# -----------------------------
# Construction (initial layout)
# -----------------------------
def golden_spiral_candidate(k: int, step: float) -> Tuple[float, float]:
    # deterministic spread-out points
    # k starts from 1
    ang = k * 2.399963229728653  # golden angle
    r = step * math.sqrt(k)
    return (r * math.cos(ang), r * math.sin(ang))


def build_initial_prefix(
    params: List[TreeParam],
    radii: np.ndarray,
    seed: int,
    step_mult: float = 2.2,
    max_tries: int = 50000,
) -> Tuple[List[Placement], np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)

    n = len(params)
    places: List[Placement] = [Placement(0.0, 0.0, 0.0) for _ in range(n)]
    verts = np.zeros((n, 3, 2), dtype=np.float64)

    scale = float(np.max(radii)) if n > 0 else 1.0
    step = step_mult * scale

    for i in range(n):
        placed = False
        for k in range(1, max_tries + 1):
            x, y = golden_spiral_candidate(k, step)
            x += random.uniform(-0.15 * step, 0.15 * step)
            y += random.uniform(-0.15 * step, 0.15 * step)
            x, y = clip_xy(x, y)

            deg = random.uniform(0.0, 180.0)
            places[i] = Placement(x, y, deg)
            verts[i] = triangle_vertices(params[i], places[i])

            # check overlap against previous
            ok = True
            for j in range(i):
                if polys_overlap_sat(verts[i], verts[j]):
                    ok = False
                    break
            if ok:
                placed = True
                break

        if not placed:
            # fallback: random search around origin
            for _ in range(20000):
                x = random.uniform(-10.0, 10.0)
                y = random.uniform(-10.0, 10.0)
                x, y = clip_xy(x, y)
                deg = random.uniform(0.0, 180.0)
                places[i] = Placement(x, y, deg)
                verts[i] = triangle_vertices(params[i], places[i])
                ok = True
                for j in range(i):
                    if polys_overlap_sat(verts[i], verts[j]):
                        ok = False
                        break
                if ok:
                    placed = True
                    break
            if not placed:
                raise RuntimeError(f"Failed to place item {i} (n={n}). Try bigger step_mult.")
    return places, verts


# -----------------------------
# Shrink
# -----------------------------
def shrink_about_bbox_center(
    params: List[TreeParam],
    places: List[Placement],
    verts: np.ndarray,
    iters: int = 25,
    shrink0: float = 0.995,
    shrink1: float = 0.975,
) -> Tuple[List[Placement], np.ndarray]:
    # gradually scale positions towards bbox center; accept only if no overlaps
    best_places = [Placement(p.x, p.y, p.deg) for p in places]
    best_verts = verts.copy()
    best_side = bbox_side(best_verts)

    for t in range(iters):
        alpha = shrink0 + (shrink1 - shrink0) * (t / max(1, iters - 1))
        cx, cy = bbox_center(best_verts)

        cand_places = []
        for p in best_places:
            nx = cx + alpha * (p.x - cx)
            ny = cy + alpha * (p.y - cy)
            nx, ny = clip_xy(nx, ny)
            cand_places.append(Placement(nx, ny, p.deg))

        cand_verts = compute_vertices(params, cand_places)
        if not any_overlap_all(cand_verts):
            cand_side = bbox_side(cand_verts)
            if cand_side <= best_side + 1e-12:
                best_side = cand_side
                best_places = cand_places
                best_verts = cand_verts
    return best_places, best_verts


# -----------------------------
# SA optimization
# -----------------------------
def pick_index_boundary_biased(verts: np.ndarray, bias_p: float = 0.85) -> int:
    # choose a polygon on bbox boundary with high probability
    n = verts.shape[0]
    if n <= 2:
        return random.randrange(n)

    xs = verts[:, :, 0]
    ys = verts[:, :, 1]
    minx = xs.min()
    maxx = xs.max()
    miny = ys.min()
    maxy = ys.max()

    boundary = []
    for i in range(n):
        v = verts[i]
        # touches bbox if any vertex is close to a boundary
        if (
            np.any(np.isclose(v[:, 0], minx, atol=1e-10))
            or np.any(np.isclose(v[:, 0], maxx, atol=1e-10))
            or np.any(np.isclose(v[:, 1], miny, atol=1e-10))
            or np.any(np.isclose(v[:, 1], maxy, atol=1e-10))
        ):
            boundary.append(i)

    if boundary and random.random() < bias_p:
        return random.choice(boundary)
    return random.randrange(n)


def recenter_layout(
    places: List[Placement], verts: np.ndarray, target_center: Tuple[float, float] = (0.0, 0.0)
) -> Tuple[List[Placement], np.ndarray]:
    # Translate all placements so bbox center moves to target_center
    cx, cy = bbox_center(verts)
    dx = target_center[0] - cx
    dy = target_center[1] - cy

    cand_places = []
    for p in places:
        nx, ny = clip_xy(p.x + dx, p.y + dy)
        cand_places.append(Placement(nx, ny, p.deg))
    cand_verts = verts.copy()
    cand_verts[:, :, 0] += dx
    cand_verts[:, :, 1] += dy
    # clipping can break the "shift exactly", so recompute if any clip happened
    if any(abs((p.x + dx) - cp.x) > 1e-12 or abs((p.y + dy) - cp.y) > 1e-12 for p, cp in zip(places, cand_places)):
        cand_verts = compute_vertices(params_global_for_recenter, cand_places)  # replaced at runtime
    return cand_places, cand_verts


# small trick: provide params via global for recenter recompute
params_global_for_recenter: List[TreeParam] = []


def sa_optimize_n(
    params: List[TreeParam],
    places: List[Placement],
    radii: np.ndarray,
    steps: int,
    seed: int,
    t0: Optional[float] = None,
    t_end: float = 1e-4,
    p_rot: float = 0.45,
    p_swap: float = 0.04,
    inward_p: float = 0.55,
    recenter_every: int = 600,
) -> Tuple[List[Placement], np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)

    n = len(params)
    verts = compute_vertices(params, places)
    assert not any_overlap_all(verts), "Initial layout overlaps"

    cur_side = bbox_side(verts)
    best_side = cur_side
    best_places = [Placement(p.x, p.y, p.deg) for p in places]
    best_verts = verts.copy()

    if t0 is None:
        # temperature scale tied to current side
        t0 = 0.03 * cur_side + 1e-6

    for it in range(steps):
        frac = 1.0 - it / max(1, steps - 1)
        T = t_end + (t0 - t_end) * frac

        # step sizes shrink over time and scale with box size
        sigma_xy = max(1e-4, 0.08 * cur_side * frac)
        sigma_deg = 45.0 * frac + 0.5

        # periodic recenter to avoid clipping degeneracy
        if recenter_every > 0 and it > 0 and it % recenter_every == 0:
            # only translate; doesn't change side but improves proposal feasibility
            cx, cy = bbox_center(verts)
            dx, dy = -cx, -cy
            # shift all
            new_places = []
            clipped = False
            for p in places:
                nx, ny = p.x + dx, p.y + dy
                nx2, ny2 = clip_xy(nx, ny)
                if abs(nx2 - nx) > 1e-12 or abs(ny2 - ny) > 1e-12:
                    clipped = True
                new_places.append(Placement(nx2, ny2, p.deg))
            if clipped:
                # recompute robustly
                global params_global_for_recenter
                params_global_for_recenter = params
                verts = compute_vertices(params, new_places)
            else:
                verts[:, :, 0] += dx
                verts[:, :, 1] += dy
            places = new_places

        if random.random() < p_swap and n >= 2:
            # swap two placements (keeps set of vertices but changes association)
            i = pick_index_boundary_biased(verts, bias_p=0.9)
            j = random.randrange(n)
            if i == j:
                continue
            cand_places = places.copy()
            cand_places[i], cand_places[j] = cand_places[j], cand_places[i]
            cand_verts = compute_vertices(params, cand_places)
            if any_overlap_all(cand_verts):
                continue
            cand_side = bbox_side(cand_verts)

            dE = cand_side - cur_side
            if dE <= 0.0 or random.random() < math.exp(-dE / max(1e-12, T)):
                places = cand_places
                verts = cand_verts
                cur_side = cand_side
        else:
            # local move
            i = pick_index_boundary_biased(verts, bias_p=0.85)
            old = places[i]

            # propose
            if random.random() < p_rot:
                nd = (old.deg + random.gauss(0.0, sigma_deg)) % 180.0
                cand = Placement(old.x, old.y, nd)
            else:
                # inward move with high probability
                if random.random() < inward_p:
                    cx, cy = bbox_center(verts)
                    vx, vy = (cx - old.x), (cy - old.y)
                    # move towards center + small noise
                    t = random.uniform(0.15, 1.0)
                    nx = old.x + t * vx + random.gauss(0.0, 0.25 * sigma_xy)
                    ny = old.y + t * vy + random.gauss(0.0, 0.25 * sigma_xy)
                else:
                    nx = old.x + random.gauss(0.0, sigma_xy)
                    ny = old.y + random.gauss(0.0, sigma_xy)
                nx, ny = clip_xy(nx, ny)
                cand = Placement(nx, ny, old.deg)

            cand_places = places.copy()
            cand_places[i] = cand
            cand_verts = verts.copy()
            cand_verts[i] = triangle_vertices(params[i], cand)

            # overlap check only vs others
            if any_overlap_idx(i, cand_verts):
                continue

            cand_side = bbox_side(cand_verts)
            dE = cand_side - cur_side

            if dE <= 0.0 or random.random() < math.exp(-dE / max(1e-12, T)):
                places = cand_places
                verts = cand_verts
                cur_side = cand_side

        if cur_side < best_side - 1e-12:
            best_side = cur_side
            best_places = [Placement(p.x, p.y, p.deg) for p in places]
            best_verts = verts.copy()

    return best_places, best_verts


# -----------------------------
# ALNS (ruin & recreate)
# -----------------------------
def alns_improve_n(
    params: List[TreeParam],
    places: List[Placement],
    radii: np.ndarray,
    seed: int,
    rounds: int = 10,
    remove_frac: float = 0.08,
    per_round_sa_steps: int = 900,
) -> Tuple[List[Placement], np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)

    n = len(params)
    verts = compute_vertices(params, places)
    best_side = bbox_side(verts)
    best_places = [Placement(p.x, p.y, p.deg) for p in places]
    best_verts = verts.copy()

    for rr in range(rounds):
        cur_places = [Placement(p.x, p.y, p.deg) for p in best_places]
        cur_verts = best_verts.copy()
        cur_side = best_side

        k = max(1, int(remove_frac * n))
        removed = set()
        # boundary-biased removals
        while len(removed) < k:
            idx = pick_index_boundary_biased(cur_verts, bias_p=0.9)
            removed.add(idx)
        removed = sorted(list(removed))

        keep = [i for i in range(n) if i not in removed]

        # build partial
        keep_places = [cur_places[i] for i in keep]
        keep_params = [params[i] for i in keep]
        keep_radii = radii[keep]
        keep_verts = compute_vertices(keep_params, keep_places)

        # reinsertion order: big items first (among removed)
        removed_sorted = sorted(removed, key=lambda i: radii[i], reverse=True)

        # start candidate layout arrays in original indexing
        cand_places = cur_places.copy()
        cand_verts_full = cur_verts.copy()

        # helper: check overlap for idx against already placed (kept + previously inserted)
        def ok_place_for(idx: int, v: np.ndarray) -> bool:
            for j in range(n):
                if j == idx:
                    continue
                # if j is not yet "active", skip
                if j in removed and j not in inserted:
                    continue
                if polys_overlap_sat(v, cand_verts_full[j]):
                    return False
            return True

        inserted = set()
        cx, cy = bbox_center(cur_verts)

        for idx in removed_sorted:
            placed = False
            # try many candidates around bbox center / spiral
            step = max(1.0, 1.8 * float(radii[idx]))
            for t in range(1, 2500):
                x0, y0 = golden_spiral_candidate(t, step)
                x = cx + x0
                y = cy + y0
                x, y = clip_xy(x, y)
                deg = random.uniform(0.0, 180.0)

                cand_p = Placement(x, y, deg)
                v = triangle_vertices(params[idx], cand_p)

                if ok_place_for(idx, v):
                    cand_places[idx] = cand_p
                    cand_verts_full[idx] = v
                    inserted.add(idx)
                    placed = True
                    break

            if not placed:
                # fallback random
                for _ in range(10000):
                    x = random.uniform(XY_MIN, XY_MAX)
                    y = random.uniform(XY_MIN, XY_MAX)
                    deg = random.uniform(0.0, 180.0)
                    cand_p = Placement(x, y, deg)
                    v = triangle_vertices(params[idx], cand_p)
                    if ok_place_for(idx, v):
                        cand_places[idx] = cand_p
                        cand_verts_full[idx] = v
                        inserted.add(idx)
                        placed = True
                        break
            if not placed:
                # give up this round
                inserted = None
                break

        if inserted is None:
            continue

        if any_overlap_all(cand_verts_full):
            continue

        # short SA polish
        cand_places, cand_verts_full = sa_optimize_n(
            params=params,
            places=cand_places,
            radii=radii,
            steps=per_round_sa_steps,
            seed=seed + 991 * rr + 7,
            p_rot=0.40,
            p_swap=0.02,
            inward_p=0.60,
            recenter_every=700,
        )

        # optional shrink
        cand_places, cand_verts_full = shrink_about_bbox_center(
            params=params, places=cand_places, verts=cand_verts_full, iters=18, shrink0=0.995, shrink1=0.980
        )

        cand_side = bbox_side(cand_verts_full)
        if cand_side < best_side - 1e-12:
            best_side = cand_side
            best_places = [Placement(p.x, p.y, p.deg) for p in cand_places]
            best_verts = cand_verts_full.copy()

    return best_places, best_verts


# -----------------------------
# Budget schedule (IMPORTANT!)
# -----------------------------
def steps_budget(n: int, base_steps: int) -> int:
    # Big-n needs MORE work; your previous schedule under-optimized n=200.
    if n <= 20:
        return int(base_steps)
    if n <= 60:
        return int(base_steps * 0.75)
    if n <= 100:
        return int(base_steps * 0.55)
    if n <= 140:
        return int(base_steps * 0.60)
    if n <= 170:
        return int(base_steps * 0.75)
    # hardest tail
    return int(base_steps * 1.10)


# -----------------------------
# Solve all prefixes 1..N
# -----------------------------
def solve_all_prefixes(
    params_all: List[TreeParam],
    seed: int,
    base_steps: int,
    attempts_per_n: int = DEFAULT_ATTEMPTS_PER_N,
    do_shrink: bool = True,
    do_alns: bool = True,
    alns_rounds_small: int = 10,
    alns_rounds_big: int = 6,
) -> Dict[int, List[Placement]]:
    N = len(params_all)
    radii_all = np.array([max(p.r, p.h) for p in params_all], dtype=np.float64)

    solutions: Dict[int, List[Placement]] = {}
    places: List[Placement] = []
    verts: Optional[np.ndarray] = None

    sides: List[float] = []
    score_so_far = 0.0

    for n in range(1, N + 1):
        params = params_all[:n]
        radii = radii_all[:n]

        best_places: Optional[List[Placement]] = None
        best_verts: Optional[np.ndarray] = None
        best_side = 1e100

        for att in range(attempts_per_n):
            # init from scratch each attempt (more robust for hard n)
            p0, v0 = build_initial_prefix(params, radii, seed=seed + 999 * n + 23 * att, step_mult=2.2)
            n_steps = steps_budget(n, base_steps)

            p1, v1 = sa_optimize_n(
                params=params,
                places=p0,
                radii=radii,
                steps=n_steps,
                seed=seed + 1337 * n + 101 * att,
                p_rot=0.45,
                p_swap=0.04,
                inward_p=0.60,
                recenter_every=650,
            )

            if do_alns and n >= 25:
                rounds = alns_rounds_small if n < 120 else alns_rounds_big
                p1, v1 = alns_improve_n(
                    params=params,
                    places=p1,
                    radii=radii,
                    seed=seed + 1777 * n + 17 * att,
                    rounds=rounds,
                    remove_frac=0.08 if n < 120 else 0.10,
                    per_round_sa_steps=700 if n < 120 else 1100,
                )

            if do_shrink and n >= 12:
                p1, v1 = shrink_about_bbox_center(params=params, places=p1, verts=v1, iters=22)

            side = bbox_side(v1)
            if side < best_side:
                best_side = side
                best_places = p1
                best_verts = v1

        assert best_places is not None and best_verts is not None
        solutions[n] = best_places
        places = best_places
        verts = best_verts

        sides.append(best_side)
        score_so_far += score_term(best_side, n)

        if n in (1, 2, 5, 10, 20, 50, 80, 100, 120, 150, 200):
            print(
                f"[n={n:3d}] side ≈ {best_side:.6f}  score_so_far ≈ {score_so_far:.6f}  steps={steps_budget(n, base_steps)}"
            )

    df = pd.DataFrame(
        {
            "n": np.arange(1, N + 1),
            "side": np.array(sides),
            "term": (np.array(sides) ** 2) / np.arange(1, N + 1),
        }
    )
    df.to_csv("debug_sides.csv", index=False)
    print(f"TOTAL score ≈ {df['term'].sum():.6f}")

    return solutions


# -----------------------------
# IO
# -----------------------------
def load_tree_params(path: str = "trees.csv") -> List[TreeParam]:
    df = pd.read_csv(path)
    # common possibilities
    cols = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for nm in names:
            if nm in cols:
                return cols[nm]
        return None

    r_col = pick("r", "radius", "base_r", "half_base", "halfwidth")
    h_col = pick("h", "height", "tree_h", "apex_h")

    if r_col is None or h_col is None:
        raise ValueError(
            f"Cannot find r/h columns in {path}. Found columns: {list(df.columns)}. "
            f"Expected something like r,h or radius,height."
        )

    params = [TreeParam(float(r), float(h)) for r, h in zip(df[r_col].values, df[h_col].values)]
    print(f"Loaded {len(params)} trees from {path} (r_col={r_col}, h_col={h_col}).")
    return params


def write_submission(solutions: Dict[int, List[Placement]], out_path: str = "submission.csv") -> None:
    rows = []
    # solutions has keys 1..N
    for n in range(1, max(solutions.keys()) + 1):
        places = solutions[n]
        for i, p in enumerate(places):
            # Kaggle format: string values prefixed with 's'
            rows.append(
                {
                    "id": f"{n:03d}_{i}",
                    "x": f"s{p.x:.10f}",
                    "y": f"s{p.y:.10f}",
                    "deg": f"s{(p.deg % 180.0):.10f}",
                }
            )
    sub = pd.DataFrame(rows, columns=["id", "x", "y", "deg"])
    sub.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  (rows={len(sub)})")


# -----------------------------
# Main: multi-seed pick best
# -----------------------------
def main():
    params_all = load_tree_params("trees.csv")
    N = len(params_all)
    assert N >= 200, "Expected at least 200 trees in trees.csv"

    best_score = 1e100
    best_seed = None
    best_solutions = None

    for seed in DEFAULT_SEEDS:
        print("=" * 70)
        print(f"RUN seed={seed}")
        sols = solve_all_prefixes(
            params_all=params_all[:200],
            seed=seed,
            base_steps=DEFAULT_BASE_STEPS,
            attempts_per_n=DEFAULT_ATTEMPTS_PER_N,
            do_shrink=True,
            do_alns=True,
            alns_rounds_small=10,
            alns_rounds_big=6,
        )
        df = pd.read_csv("debug_sides.csv")
        total = float(df["term"].sum())
        print(f"seed={seed} TOTAL={total:.6f}")

        if total < best_score:
            best_score = total
            best_seed = seed
            best_solutions = sols

    print("=" * 70)
    print(f"BEST seed={best_seed} score={best_score:.6f}")
    assert best_solutions is not None
    write_submission(best_solutions, "submission.csv")


if __name__ == "__main__":
    main()
