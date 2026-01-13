"""JAX collision predicates for polygon intersection.

These functions are used by the JAX SA optimizer as a fine collision check.
They intentionally trade absolute geometric rigor for speed and JIT-friendliness.
For submission validation and local scoring, prefer the NumPy-based predicates in
`santa_packing.scoring`.
"""

import jax
import jax.numpy as jnp

from .constants import EPS


@jax.jit
def cross_product(o, a, b):
    """2D cross product z-component: `(a - o) x (b - o)`."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


@jax.jit
def segments_intersect(p1, p2, p3, p4):
    """Check if two segments strictly intersect (touching excluded).

    Args:
        p1: Segment 1 start `(2,)`.
        p2: Segment 1 end `(2,)`.
        p3: Segment 2 start `(2,)`.
        p4: Segment 2 end `(2,)`.

    Returns:
        Boolean JAX scalar.
    """
    # Using orientation test (cross products)
    d1 = cross_product(p3, p4, p1)
    d2 = cross_product(p3, p4, p2)
    d3 = cross_product(p1, p2, p3)
    d4 = cross_product(p1, p2, p4)

    # Check signs
    intersect_strict = ((d1 > EPS) != (d2 > EPS)) & ((d3 > EPS) != (d4 > EPS))

    # Consider collinearity/touching if needed, but for strict overlap we usually want > 0
    # For now, strict intersection
    return intersect_strict


@jax.jit
def point_in_polygon(point, poly):
    """Return True if a point is inside a polygon (ray casting).

    Args:
        point: Point `(2,)`.
        poly: Polygon vertices `(V, 2)`.

    Returns:
        Boolean JAX scalar.
    """
    x, y = point
    n = poly.shape[0]

    # JAX scan or vectorized check for edges
    # We check how many times a ray to the right intersects edges

    def body_fun(carry, i):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]

        # Check if edge intersects ray from (x,y) to (inf, y)
        # Conditions:
        # 1. One point above/on_line y, one below/on_line y
        # 2. Intersection x > point.x

        cond1 = (p1[1] > y) != (p2[1] > y)

        # x coordinate of intersection of line p1-p2 with y=point.y
        # x_int = p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y)
        # We want x_int > x

        # Avoid division by zero: if p1[1] == p2[1], cond1 is false anyway
        x_int = (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1] + EPS) + p1[0]

        cond2 = x + EPS < x_int

        intersect = cond1 & cond2
        return carry ^ intersect, None

    inside, _ = jax.lax.scan(body_fun, False, jnp.arange(n))
    return inside


@jax.jit
def polygons_intersect(poly1, poly2):
    """Check if two polygons intersect.

    Method:
    1. AABB overlap fast reject.
    2. Edge-edge intersection checks.
    3. Point-in-polygon inclusion tests (either direction).

    Args:
        poly1: Vertices `(V1, 2)`.
        poly2: Vertices `(V2, 2)`.

    Returns:
        Boolean JAX scalar.
    """

    # 1. Bounding Box Check (fast reject)
    min1 = jnp.min(poly1, axis=0)
    max1 = jnp.max(poly1, axis=0)
    min2 = jnp.min(poly2, axis=0)
    max2 = jnp.max(poly2, axis=0)

    bbox_overlap = jnp.all(max1 >= min2) & jnp.all(max2 >= min1)

    def _expensive() -> jax.Array:
        # 2. Edge Intersections (NxM checks; here N=M=15 => 225)
        n1 = poly1.shape[0]
        n2 = poly2.shape[0]

        edges1_start = poly1
        edges1_end = jnp.roll(poly1, -1, axis=0)

        edges2_start = poly2
        edges2_end = jnp.roll(poly2, -1, axis=0)

        def check_edges(i, j):
            return segments_intersect(edges1_start[i], edges1_end[i], edges2_start[j], edges2_end[j])

        edge_overlaps = jax.vmap(lambda i: jax.vmap(lambda j: check_edges(i, j))(jnp.arange(n2)))(jnp.arange(n1))
        has_edge_overlap = jnp.any(edge_overlaps)

        # 3. Vertex Inclusion (Poly1 in Poly2)
        verts_in_poly2 = jax.vmap(lambda p: point_in_polygon(p, poly2))(poly1)
        has_v1_in_p2 = jnp.any(verts_in_poly2)

        # 4. Vertex Inclusion (Poly2 in Poly1)
        verts_in_poly1 = jax.vmap(lambda p: point_in_polygon(p, poly1))(poly2)
        has_v2_in_p1 = jnp.any(verts_in_poly1)

        return has_edge_overlap | has_v1_in_p2 | has_v2_in_p1

    return jax.lax.cond(bbox_overlap, _expensive, lambda: jnp.array(False))
