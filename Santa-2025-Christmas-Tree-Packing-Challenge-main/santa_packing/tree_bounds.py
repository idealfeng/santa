"""Fast conservative bounds for the tree polygon.

This module precomputes per-angle axis-aligned bounding boxes (AABBs) for the
single polygon used in the repo (the Santa 2025 tree). The SA optimizer uses
these AABBs as a cheap broad-phase collision filter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .tree import get_tree_polygon


def polygon_radius2(poly: jax.Array) -> jax.Array:
    """Squared enclosing-circle radius around the origin for a polygon."""
    return jnp.max(jnp.sum(poly * poly, axis=1))


def precompute_aabb_table(poly: jax.Array, *, step_deg: float) -> jax.Array:
    """Precompute axis-aligned bounding boxes for rotations of `poly`.

    Returns:
        (B, 4) array where each row is [min_x, min_y, max_x, max_y] for angle = b*step_deg.
    """
    step = jnp.asarray(step_deg, dtype=poly.dtype)
    bins = int(round(360.0 / float(step_deg)))
    if abs(bins * float(step_deg) - 360.0) > 1e-6:
        raise ValueError(f"step_deg={step_deg} must divide 360 exactly")

    angles_deg = jnp.arange(bins, dtype=poly.dtype) * step
    angles_rad = jnp.deg2rad(angles_deg)
    c = jnp.cos(angles_rad)[:, None]
    s = jnp.sin(angles_rad)[:, None]

    x = poly[:, 0][None, :]
    y = poly[:, 1][None, :]
    xr = x * c - y * s
    yr = x * s + y * c

    min_x = jnp.min(xr, axis=1)
    max_x = jnp.max(xr, axis=1)
    min_y = jnp.min(yr, axis=1)
    max_y = jnp.max(yr, axis=1)
    return jnp.stack([min_x, min_y, max_x, max_y], axis=1)


# Tree constants (the only polygon in this repo)
TREE_POLY = get_tree_polygon()

# Bounding circle (rotation-invariant)
TREE_RADIUS2 = polygon_radius2(TREE_POLY)
TREE_RADIUS = jnp.sqrt(TREE_RADIUS2)

# AABB lookup (rotation-dependent, approximated by binning theta in degrees)
TREE_AABB_STEP_DEG = 1.0
TREE_AABB_BINS = int(round(360.0 / TREE_AABB_STEP_DEG))
TREE_AABB_TABLE = precompute_aabb_table(TREE_POLY, step_deg=TREE_AABB_STEP_DEG)

# Padding to make binned AABB conservative for any theta within ±step/2.
_HALF_STEP_RAD = jnp.deg2rad(jnp.asarray(TREE_AABB_STEP_DEG * 0.5, dtype=TREE_POLY.dtype))
TREE_AABB_PAD = TREE_RADIUS * _HALF_STEP_RAD + jnp.asarray(1e-6, dtype=TREE_POLY.dtype)
TREE_AABB_TABLE_PADDED = TREE_AABB_TABLE + jnp.array(
    [-TREE_AABB_PAD, -TREE_AABB_PAD, TREE_AABB_PAD, TREE_AABB_PAD],
    dtype=TREE_POLY.dtype,
)


def theta_to_aabb_bin(theta_deg: jax.Array) -> jax.Array:
    """Map a rotation angle in degrees to the closest precomputed AABB bin."""
    idx = jnp.rint(theta_deg / TREE_AABB_STEP_DEG).astype(jnp.int32)
    return jnp.mod(idx, TREE_AABB_BINS)


def aabb_for_poses(poses: jax.Array, *, padded: bool = False) -> jax.Array:
    """Return per-pose AABBs in world coordinates as (N, 4)."""
    table = TREE_AABB_TABLE_PADDED if padded else TREE_AABB_TABLE
    idx = theta_to_aabb_bin(poses[:, 2])
    local = table[idx]
    xy = poses[:, 0:2]
    xyxy = jnp.concatenate([xy, xy], axis=1)
    return local + xyxy


def aabb_overlap(a: jax.Array, b: jax.Array) -> jax.Array:
    """Return True if two AABBs [minx,miny,maxx,maxy] overlap."""
    return (a[2] >= b[0]) & (b[2] >= a[0]) & (a[3] >= b[1]) & (b[3] >= a[1])
