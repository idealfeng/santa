"""JAX geometry helpers for rigid transforms.

This module implements basic operations used by the JAX-based optimizer:
rotation/translation of points and polygons, plus AABB extraction.

Representation conventions:
- A point is a `(2,)` array.
- A polygon is a `(V, 2)` array of vertices.
- A pose is a `(3,)` array: `[x, y, theta_degrees]`.
"""

import jax.numpy as jnp


def rotate_point(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    """Rotate a point around the origin.

    Args:
        p: Point `(2,)`.
        theta: Rotation angle in radians.

    Returns:
        Rotated point `(2,)`.
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    rotation_matrix = jnp.array([[c, -s], [s, c]])
    return jnp.dot(rotation_matrix, p)


def rotate_polygon(poly: jnp.ndarray, theta: float) -> jnp.ndarray:
    """Rotate a polygon around the origin.

    Args:
        poly: Polygon vertices `(V, 2)`.
        theta: Rotation angle in radians.

    Returns:
        Rotated polygon vertices `(V, 2)`.
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    # Rotation matrix shape (2, 2)
    rotation_matrix = jnp.array([[c, -s], [s, c]])
    # poly shape (N, 2), result (N, 2)
    return jnp.dot(poly, rotation_matrix.T)


def translate_polygon(poly: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Translate a polygon by (dx, dy).

    Args:
        poly: Polygon vertices `(V, 2)`.
        dx: X translation.
        dy: Y translation.

    Returns:
        Translated polygon vertices `(V, 2)`.
    """
    translation = jnp.array([dx, dy])
    return poly + translation


def transform_polygon(poly: jnp.ndarray, pose: jnp.ndarray) -> jnp.ndarray:
    """Apply a rigid transform (rotation + translation) to a polygon.

    Args:
        poly: Polygon vertices `(V, 2)`.
        pose: Pose `(3,)` as `[x, y, theta_degrees]`.

    Returns:
        Transformed polygon vertices `(V, 2)`.
    """
    x, y, theta_deg = pose
    theta_rad = jnp.deg2rad(theta_deg)

    rotated = rotate_polygon(poly, theta_rad)
    translated = translate_polygon(rotated, x, y)
    return translated


def polygon_bbox(poly: jnp.ndarray) -> jnp.ndarray:
    """Compute the axis-aligned bounding box (AABB) of a polygon.

    Args:
        poly: Polygon vertices `(V, 2)`.

    Returns:
        A `(4,)` array `[min_x, min_y, max_x, max_y]`.
    """
    min_vals = jnp.min(poly, axis=0)  # (2,)
    max_vals = jnp.max(poly, axis=0)  # (2,)
    return jnp.concatenate([min_vals, max_vals])
