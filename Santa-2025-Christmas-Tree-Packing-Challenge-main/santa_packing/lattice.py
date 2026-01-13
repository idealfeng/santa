"""Deterministic lattice-based initial layouts (NumPy).

`lattice_poses` is a fast, reproducible baseline that places tree centers on a
square or hexagonal lattice with optional rotation patterns. The resulting
poses are shifted so the packing starts near the origin, which tends to help
small `n` instances.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal, Sequence

import numpy as np

from .geom_np import polygon_radius, shift_poses_to_origin, transform_polygon
from .scoring import polygons_intersect_strict
from .tree_data import TREE_POINTS

Pattern = Literal["hex", "square"]
RotateMode = Literal["constant", "row", "checker", "ring"]

_TREE_POINTS_NP = np.array(TREE_POINTS, dtype=float)


def _normalize_rotate_seq(rotate_deg: float, rotate_degs: Sequence[float] | None) -> tuple[float, ...]:
    if rotate_degs is None:
        return (float(rotate_deg),)
    seq = [float(x) for x in rotate_degs if np.isfinite(x)]
    return tuple(seq) if seq else (float(rotate_deg),)


def _cell_rotation(
    row: int,
    col: int,
    *,
    rows: int,
    cols: int,
    rotate_mode: RotateMode,
    rotate_seq: tuple[float, ...],
) -> float:
    if rotate_mode == "constant" or len(rotate_seq) == 1:
        return float(rotate_seq[0])
    if rotate_mode == "row":
        return float(rotate_seq[row % len(rotate_seq)])
    if rotate_mode == "checker":
        return float(rotate_seq[(row + col) % len(rotate_seq)])
    if rotate_mode == "ring":
        center_row = 0.5 * float(rows - 1)
        center_col = 0.5 * float(cols - 1)
        ring = int(max(abs(float(row) - center_row), abs(float(col) - center_col)))
        return float(rotate_seq[ring % len(rotate_seq)])
    raise ValueError(f"Unknown rotate_mode={rotate_mode!r}")


def lattice_poses(
    n: int,
    *,
    pattern: Pattern = "hex",
    margin: float = 0.02,
    rotate_deg: float = 0.0,
    rotate_mode: RotateMode = "constant",
    rotate_degs: Sequence[float] | None = None,
) -> np.ndarray:
    """Generate a lattice packing for `n` trees.

    Args:
        n: Number of trees.
        pattern: `"hex"` or `"square"`.
        margin: Extra spacing fraction used during spacing search (robustness).
        rotate_deg: Base rotation in degrees (used when `rotate_degs` is not set).
        rotate_mode: Rotation assignment strategy across the lattice.
        rotate_degs: Optional sequence of rotations in degrees used by non-constant modes.

    Returns:
        Array `(n, 3)` with `[x, y, theta_deg]` poses shifted to the origin.
    """
    points = _TREE_POINTS_NP
    if rotate_mode == "constant":
        rotate_seq = (float(rotate_deg),)
    else:
        rotate_seq = _normalize_rotate_seq(rotate_deg, rotate_degs)
    step, row_height = _compute_spacing_cached(pattern, rotate_mode, rotate_seq, margin)

    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    poses = np.zeros((n, 3), dtype=float)

    if pattern == "hex":
        for i in range(n):
            row = i // cols
            col = i % cols
            x = col * step + (step / 2.0 if row % 2 == 1 else 0.0)
            y = row * row_height
            theta = _cell_rotation(row, col, rows=rows, cols=cols, rotate_mode=rotate_mode, rotate_seq=rotate_seq)
            poses[i] = (x, y, theta)
    else:
        for i in range(n):
            row = i // cols
            col = i % cols
            x = col * step
            y = row * row_height
            theta = _cell_rotation(row, col, rows=rows, cols=cols, rotate_mode=rotate_mode, rotate_seq=rotate_seq)
            poses[i] = (x, y, theta)

    return shift_poses_to_origin(points, poses)


@lru_cache(maxsize=None)
def _compute_spacing_cached(
    pattern: Pattern,
    rotate_mode: RotateMode,
    rotate_seq: tuple[float, ...],
    margin: float,
) -> tuple[float, float]:
    return _compute_spacing(
        _TREE_POINTS_NP,
        pattern,
        rotate_deg=float(rotate_seq[0]),
        margin=margin,
        rotate_mode=rotate_mode,
        rotate_degs=rotate_seq,
    )


def _compute_spacing(
    points: np.ndarray,
    pattern: Pattern,
    rotate_deg: float,
    margin: float,
    *,
    rotate_mode: RotateMode = "constant",
    rotate_degs: Sequence[float] | None = None,
) -> tuple[float, float]:
    radius = polygon_radius(points)
    upper = 2.5 * radius
    margin_eff = float(margin)
    if margin_eff < 0.0:
        margin_eff = 0.0
    # Even with strict collision checks, we need a tiny gap for fp robustness.
    margin_eff = max(margin_eff, 1e-8)

    if rotate_mode == "constant":
        rotate_seq = (float(rotate_deg),)
    else:
        rotate_seq = _normalize_rotate_seq(rotate_deg, rotate_degs)

    # For spacing search we allow "touch" (strict) and rely on a (tiny) margin for a gap.
    intersects = polygons_intersect_strict

    def _patch_dims() -> tuple[int, int]:
        if rotate_mode == "ring":
            side = max(3, 2 * len(rotate_seq) + 1)
            return side, side

        seq_len = len(rotate_seq)
        if rotate_mode == "constant" or seq_len <= 1:
            period_rows = 1
            period_cols = 1
        elif rotate_mode == "row":
            period_rows = math.lcm(seq_len, 2) if pattern == "hex" else seq_len
            period_cols = 1
        elif rotate_mode == "checker":
            period_rows = math.lcm(seq_len, 2) if pattern == "hex" else seq_len
            period_cols = seq_len
        else:
            raise ValueError(f"Unknown rotate_mode={rotate_mode!r}")

        rows = max(2, int(period_rows) + 1)
        cols = max(2, int(period_cols) + 1)
        return rows, cols

    patch_rows, patch_cols = _patch_dims()

    def _patch_ok(step: float, row_height: float) -> bool:
        rows = patch_rows
        cols = patch_cols
        polys: list[list[np.ndarray]] = [[None for _ in range(cols)] for _ in range(rows)]  # type: ignore[list-item]
        for r in range(rows):
            for c in range(cols):
                if pattern == "hex":
                    x = float(c) * step + (step / 2.0 if (r % 2 == 1) else 0.0)
                    y = float(r) * row_height
                else:
                    x = float(c) * step
                    y = float(r) * row_height
                theta = _cell_rotation(
                    r,
                    c,
                    rows=rows,
                    cols=cols,
                    rotate_mode=rotate_mode,
                    rotate_seq=rotate_seq,
                )
                polys[r][c] = transform_polygon(points, np.array([x, y, theta], dtype=float))

        if pattern == "square":
            for r in range(rows):
                for c in range(cols):
                    p = polys[r][c]
                    if c + 1 < cols and intersects(p, polys[r][c + 1]):
                        return False
                    if r + 1 < rows:
                        if intersects(p, polys[r + 1][c]):
                            return False
                        if c + 1 < cols and intersects(p, polys[r + 1][c + 1]):
                            return False
                        if c - 1 >= 0 and intersects(p, polys[r + 1][c - 1]):
                            return False
            return True

        # Hex neighbor set: right + 2 down-diagonals (depends on row parity).
        for r in range(rows):
            for c in range(cols):
                p = polys[r][c]
                if c + 1 < cols and intersects(p, polys[r][c + 1]):
                    return False
                if r + 1 >= rows:
                    continue
                if intersects(p, polys[r + 1][c]):
                    return False
                if r % 2 == 0:
                    if c - 1 >= 0 and intersects(p, polys[r + 1][c - 1]):
                        return False
                else:
                    if c + 1 < cols and intersects(p, polys[r + 1][c + 1]):
                        return False
        return True

    def _binary_search_min(low: float, high: float, ok_fn) -> float:
        low = float(low)
        high = float(high)
        if not ok_fn(high):
            for scale in (2.0, 3.0, 5.0, 8.0):
                cand = high * scale
                if ok_fn(cand):
                    high = cand
                    break
        for _ in range(50):
            mid = (low + high) / 2.0
            if ok_fn(mid):
                high = mid
            else:
                low = mid
        return float(high)

    far = max(upper, 3.0 * radius)

    if pattern == "hex":
        dx = _binary_search_min(0.0, upper, lambda v: _patch_ok(float(v), far))
        row_height = _binary_search_min(0.0, upper, lambda v: _patch_ok(dx, float(v)))
        row_height = max(float(row_height), 1e-6)
    else:
        dx0 = _binary_search_min(0.0, upper, lambda v: _patch_ok(float(v), far))
        dy0 = _binary_search_min(0.0, upper, lambda v: _patch_ok(far, float(v)))

        def _scale_ok(scale: float) -> bool:
            return _patch_ok(dx0 * float(scale), dy0 * float(scale))

        scale = _binary_search_min(1.0, 3.0, _scale_ok)
        cand = [(dx0 * scale, dy0 * scale)]

        dy_fix = _binary_search_min(dy0, upper, lambda v: _patch_ok(dx0, float(v)))
        cand.append((dx0, dy_fix))

        dx_fix = _binary_search_min(dx0, upper, lambda v: _patch_ok(float(v), dy0))
        cand.append((dx_fix, dy0))

        dx, row_height = min(cand, key=lambda t: max(float(t[0]), float(t[1])))

    step = float(dx) * (1.0 + margin_eff)
    row_height = float(row_height) * (1.0 + margin_eff)
    return step, row_height
