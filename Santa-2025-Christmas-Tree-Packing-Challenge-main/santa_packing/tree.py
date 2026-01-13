"""Tree shape utilities."""

import jax.numpy as jnp

from .tree_data import TREE_POINTS


def get_tree_polygon() -> jnp.ndarray:
    """Return the static 15-vertex polygon defining the Christmas tree.

    Returns:
        A `(15, 2)` JAX array of vertices in counter-clockwise order.
    """
    # Coordinates taken from src/include/santa2025/tree_polygon.hpp
    return jnp.array(TREE_POINTS)
