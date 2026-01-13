"""NumPy geometry utilities.

These helpers mirror a subset of the JAX geometry/packing logic, but operate in
NumPy for CLI tools and local scoring.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def transform_polygon(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a rigid transform (rotation + translation) to polygon points.

    Args:
        points: Local vertices `(V, 2)`.
        pose: Pose `(3,)` as `[x, y, theta_deg]`.

    Returns:
        Transformed vertices `(V, 2)`.
    """
    x, y, theta_deg = pose
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rot.T + np.array([x, y], dtype=float)


def polygon_bbox(poly: np.ndarray) -> np.ndarray:
    """Compute the axis-aligned bounding box (AABB) of a polygon.

    Args:
        poly: Vertices `(V, 2)`.

    Returns:
        A `(4,)` array `[min_x, min_y, max_x, max_y]`.
    """
    min_xy = np.min(poly, axis=0)
    max_xy = np.max(poly, axis=0)
    return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=float)


def packing_bbox(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Compute the global AABB of a packing (all transformed polygons).

    Args:
        points: Local polygon vertices `(V, 2)`.
        poses: Poses `(N, 3)` as `[x, y, theta_deg]` per tree.

    Returns:
        A `(4,)` array `[min_x, min_y, max_x, max_y]`.
    """
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for pose in poses:
        poly = transform_polygon(points, pose)
        bbox = polygon_bbox(poly)
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])
    return np.array([min_x, min_y, max_x, max_y], dtype=float)


def packing_score(points: np.ndarray, poses: np.ndarray) -> float:
    """Compute the packing score for a single instance.

    The score is the side length of the smallest axis-aligned square that
    contains the packing, i.e. `max(width, height)` of the global AABB.

    Args:
        points: Local polygon vertices `(V, 2)`.
        poses: Poses `(N, 3)` as `[x, y, theta_deg]` per tree.

    Returns:
        The scalar packing score `s` for this `N`.
    """
    bbox = packing_bbox(points, poses)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return float(max(width, height))


def shift_poses_to_origin(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Translate poses so the packing's AABB starts at the origin (min_x=min_y=0).

    Args:
        points: Local polygon vertices `(V, 2)`.
        poses: Poses `(N, 3)`.

    Returns:
        A copy of poses translated in x/y.
    """
    bbox = packing_bbox(points, poses)
    shift_x = -bbox[0]
    shift_y = -bbox[1]
    shifted = np.array(poses, dtype=float, copy=True)
    shifted[:, 0] += shift_x
    shifted[:, 1] += shift_y
    return shifted


def polygon_radius(points: np.ndarray) -> float:
    """Return the radius of the smallest circle centered at origin containing all vertices."""
    norms = np.linalg.norm(points, axis=1)
    return float(np.max(norms))


def prefix_score(s_values: Iterable[float]) -> float:
    """Compute the official prefix objective from `s_1..s_N`.

    Args:
        s_values: Iterable of per-prefix packing scores `s_n`.

    Returns:
        `sum_{n=1..N} s_n^2 / n`.
    """
    total = 0.0
    for idx, s in enumerate(s_values, start=1):
        total += (s * s) / idx
    return total
