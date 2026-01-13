"""Collision checks for JAX packings.

The functions in this module combine coarse filters (bounding circle and AABB)
with an exact polygon intersection predicate to keep SA iterations fast.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import transform_polygon
from .physics import polygons_intersect
from .tree_bounds import TREE_RADIUS2, aabb_for_poses


def check_any_collisions(poses: jax.Array, base_poly: jax.Array) -> jax.Array:
    """Return True if any pair of polygons intersects.

    Notes:
    - This is still O(N^2) overall, but uses cheap coarse filters (circle + AABB)
      to avoid most expensive polygon intersection checks.
    - For SA steps that move a single index, prefer `check_collision_for_index`.
    """

    n = poses.shape[0]
    centers = poses[:, :2]

    # Coarse filter: bounding circle (squared distances).
    thr2 = 4.0 * TREE_RADIUS2 * (1.0 + 1e-4)
    d = centers[:, None, :] - centers[None, :, :]
    dist2 = jnp.sum(d * d, axis=-1)
    candidate = dist2 <= thr2

    # Coarse filter: conservative AABB overlap (uses binned theta + padding).
    bboxes = aabb_for_poses(poses, padded=True)
    bbox_overlap = (
        (bboxes[:, None, 2] >= bboxes[None, :, 0])
        & (bboxes[None, :, 2] >= bboxes[:, None, 0])
        & (bboxes[:, None, 3] >= bboxes[None, :, 1])
        & (bboxes[None, :, 3] >= bboxes[:, None, 1])
    )
    candidate = candidate & bbox_overlap

    # Only check upper triangle (unique pairs).
    candidate = jnp.triu(candidate, k=1)

    polys = jax.vmap(lambda p: transform_polygon(base_poly, p))(poses)

    def check_pair(i, j):
        return jax.lax.cond(
            candidate[i, j],
            lambda: polygons_intersect(polys[i], polys[j]),
            lambda: jnp.array(False),
        )

    matrix = jax.vmap(lambda i: jax.vmap(lambda j: check_pair(i, j))(jnp.arange(n)))(jnp.arange(n))
    return jnp.any(matrix)


def check_collision_for_index(poses: jax.Array, base_poly: jax.Array, idx: jax.Array) -> jax.Array:
    """Return True if polygon `idx` intersects any other polygon.

    This is O(N) (one-vs-all), which is much faster than checking all pairs when
    only a single tree is moved.
    """

    n = poses.shape[0]
    centers = poses[:, :2]
    center_k = centers[idx]

    # Coarse filter: if the distance between centers is greater than 2x the
    # polygon's enclosing-circle radius, intersection is impossible.
    #
    # Use squared distances to avoid sqrt (and to match `geom_np.polygon_radius`).
    thr2 = 4.0 * TREE_RADIUS2 * (1.0 + 1e-4)  # tiny safety margin for fp rounding

    d = centers - center_k
    dist2 = jnp.sum(d * d, axis=1)
    candidate = (dist2 <= thr2) & (jnp.arange(n) != idx)

    # Second coarse filter: conservative AABB overlap (uses binned theta + padding).
    bboxes = aabb_for_poses(poses, padded=True)
    bbox_k = bboxes[idx]
    bbox_overlap = (
        (bbox_k[2] >= bboxes[:, 0])
        & (bboxes[:, 2] >= bbox_k[0])
        & (bbox_k[3] >= bboxes[:, 1])
        & (bboxes[:, 3] >= bbox_k[1])
    )
    candidate = candidate & bbox_overlap

    def _check_candidates() -> jax.Array:
        poly_k = transform_polygon(base_poly, poses[idx])

        def _check_one(pose_j: jax.Array, do_test: jax.Array) -> jax.Array:
            return jax.lax.cond(
                do_test,
                lambda: polygons_intersect(poly_k, transform_polygon(base_poly, pose_j)),
                lambda: jnp.array(False),
            )

        hits = jax.vmap(_check_one)(poses, candidate)
        return jnp.any(hits)

    return jax.lax.cond(jnp.any(candidate), _check_candidates, lambda: jnp.array(False))
