"""JAX packing objectives.

This module provides packing objective functions used by the JAX SA optimizer:
- per-instance packing score (`max(width, height)` of the global AABB)
- prefix objective for a single ordered packing ("mother-prefix" use cases)
"""

import jax
import jax.numpy as jnp

from .tree_bounds import aabb_for_poses


def compute_packing_bbox(poses):
    """Compute the global packing AABB from poses.

    Args:
        poses: (N, 3) array of [x, y, theta]

    Returns:
        (4,) array [min_x, min_y, max_x, max_y]
    """
    bboxes = aabb_for_poses(poses, padded=False)

    # Global bbox
    min_x = jnp.min(bboxes[:, 0])
    min_y = jnp.min(bboxes[:, 1])
    max_x = jnp.max(bboxes[:, 2])
    max_y = jnp.max(bboxes[:, 3])

    return jnp.array([min_x, min_y, max_x, max_y])


def compute_packing_bbox_from_bboxes(bboxes: jax.Array) -> jax.Array:
    """Compute global packing AABB from per-tree AABBs.

    Args:
        bboxes: Array `(N, 4)` with `[min_x, min_y, max_x, max_y]` per tree.

    Returns:
        A `(4,)` array `[min_x, min_y, max_x, max_y]`.
    """
    min_x = jnp.min(bboxes[:, 0])
    min_y = jnp.min(bboxes[:, 1])
    max_x = jnp.max(bboxes[:, 2])
    max_y = jnp.max(bboxes[:, 3])
    return jnp.array([min_x, min_y, max_x, max_y], dtype=bboxes.dtype)


def packing_score_from_bboxes(bboxes: jax.Array) -> jax.Array:
    """Packing score from per-tree AABBs.

    Args:
        bboxes: Array `(N, 4)` with `[min_x, min_y, max_x, max_y]` per tree.

    Returns:
        Scalar `s = max(width, height)` where width/height come from the global AABB.
    """
    bbox = compute_packing_bbox_from_bboxes(bboxes)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return jnp.maximum(width, height)


def packing_score(poses):
    """Objective function for a single instance.

    Args:
        poses: Array `(N, 3)` of `[x, y, theta_deg]`.

    Returns:
        Scalar packing score `s = max(width, height)` of the global AABB.
    """
    bbox = compute_packing_bbox(poses)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return jnp.maximum(width, height)


def prefix_score(s_values):
    """Prefix-style objective: `sum_{n=1..N} s_n^2 / n`.

    Args:
        s_values: Array-like of per-prefix packing scores.

    Returns:
        Scalar prefix objective value.
    """
    s_values = jnp.array(s_values)
    n = jnp.arange(1, s_values.shape[0] + 1)
    return jnp.sum((s_values**2) / n)


def prefix_packing_score(poses):
    """Prefix objective over a single ordered packing.

    This computes the objective for a "solution mother" where the first `n`
    trees define the solution for puzzle `n`.

    Args:
        poses: Array `(N, 3)` of poses ordered by inclusion.

    Returns:
        Scalar prefix objective value.
    """
    bboxes = aabb_for_poses(poses, padded=False)
    return prefix_packing_score_from_bboxes(bboxes)


def prefix_packing_score_from_bboxes(bboxes: jax.Array) -> jax.Array:
    """Prefix objective from per-tree AABBs.

    Args:
        bboxes: Array `(N, 4)` with `[min_x, min_y, max_x, max_y]` per tree, ordered.

    Returns:
        Scalar prefix objective value.
    """

    def scan_fn(carry, bbox):
        min_x, min_y, max_x, max_y = carry
        min_x = jnp.minimum(min_x, bbox[0])
        min_y = jnp.minimum(min_y, bbox[1])
        max_x = jnp.maximum(max_x, bbox[2])
        max_y = jnp.maximum(max_y, bbox[3])
        new_carry = (min_x, min_y, max_x, max_y)
        return new_carry, jnp.array([min_x, min_y, max_x, max_y])

    if bboxes.shape[0] == 0:
        return jnp.array(0.0)
    if bboxes.shape[0] == 1:
        width = bboxes[0, 2] - bboxes[0, 0]
        height = bboxes[0, 3] - bboxes[0, 1]
        return jnp.maximum(width, height) ** 2

    init = (bboxes[0, 0], bboxes[0, 1], bboxes[0, 2], bboxes[0, 3])
    _, prefix = jax.lax.scan(scan_fn, init, bboxes[1:])
    prefix_bboxes = jnp.vstack([bboxes[0], prefix])
    widths = prefix_bboxes[:, 2] - prefix_bboxes[:, 0]
    heights = prefix_bboxes[:, 3] - prefix_bboxes[:, 1]
    s_values = jnp.maximum(widths, heights)
    return prefix_score(s_values)
