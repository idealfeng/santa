"""Submission formatting helpers.

The Santa 2025 competition expects CSV values as strings with a leading prefix
(e.g. `s0.0`) and coordinates constrained to `[-100, 100]`.
"""

from __future__ import annotations

import numpy as np

from .constants import EPS, SUBMISSION_DECIMALS, SUBMISSION_PREFIX, XY_LIMIT


def format_submission_value(value: float) -> str:
    """Format a float value as required by the competition (prefix + fixed decimals).

    Args:
        value: Numeric value to be formatted.

    Returns:
        A string like `s0.12300000000000000`.

    Normalizes -0.0 to 0.0 (via EPS) for robustness.
    """
    v = float(value)
    if abs(v) < EPS:
        v = 0.0
    return f"{SUBMISSION_PREFIX}{v:.{SUBMISSION_DECIMALS}f}"


def fit_xy_in_bounds(poses: np.ndarray) -> np.ndarray:
    """Translate poses so all x/y fit in `[-XY_LIMIT, +XY_LIMIT]` (with EPS margin).

    Args:
        poses: Array `(N, 3)` of `[x, y, theta_deg]`.

    Returns:
        A copy of `poses` translated in x/y to satisfy the constraints.

    Raises:
        ValueError: If the packing cannot fit in bounds even after translation.
    """
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses

    min_x = float(np.min(poses[:, 0]))
    max_x = float(np.max(poses[:, 0]))
    min_y = float(np.min(poses[:, 1]))
    max_y = float(np.max(poses[:, 1]))

    lo_x = (-XY_LIMIT + EPS) - min_x
    hi_x = (XY_LIMIT - EPS) - max_x
    lo_y = (-XY_LIMIT + EPS) - min_y
    hi_y = (XY_LIMIT - EPS) - max_y
    if lo_x > hi_x or lo_y > hi_y:
        raise ValueError(f"Packing does not fit in bounds: x=[{min_x:.6g},{max_x:.6g}] y=[{min_y:.6g},{max_y:.6g}]")

    poses[:, 0] += 0.5 * (lo_x + hi_x)
    poses[:, 1] += 0.5 * (lo_y + hi_y)
    return poses


def quantize_for_submission(poses: np.ndarray) -> np.ndarray:
    """Round and clamp poses for CSV emission.

    This applies:
    - rounding to `SUBMISSION_DECIMALS`
    - clamping x/y into `[-XY_LIMIT, +XY_LIMIT]` (with EPS margin)
    - normalizing near-zero values to exactly 0.0
    - wrapping degrees into `[0, 360)`

    Args:
        poses: Array `(N, 3)` of `[x, y, theta_deg]`.

    Returns:
        A quantized copy of `poses`.
    """
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses

    poses[:, 0:2] = np.round(poses[:, 0:2], SUBMISSION_DECIMALS)
    poses[:, 2] = np.round(poses[:, 2], SUBMISSION_DECIMALS)

    poses[:, 0] = np.clip(poses[:, 0], -XY_LIMIT + EPS, XY_LIMIT - EPS)
    poses[:, 1] = np.clip(poses[:, 1], -XY_LIMIT + EPS, XY_LIMIT - EPS)
    poses[np.abs(poses) < EPS] = 0.0

    poses[:, 2] = np.mod(poses[:, 2], 360.0)
    poses[np.abs(poses) < EPS] = 0.0
    return poses
