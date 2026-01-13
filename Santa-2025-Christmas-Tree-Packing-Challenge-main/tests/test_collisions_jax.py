import pytest

try:
    import jax.numpy as jnp

    from santa_packing.collisions import check_any_collisions, check_collision_for_index
    from santa_packing.tree import get_tree_polygon
except Exception:  # pragma: no cover
    pytest.skip("JAX not available", allow_module_level=True)


def test_check_any_collisions_detects_overlap() -> None:
    base_poly = get_tree_polygon()
    poses = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert bool(check_any_collisions(poses, base_poly).item()) is True


def test_check_any_collisions_no_overlap_when_far_apart() -> None:
    base_poly = get_tree_polygon()
    poses = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    assert bool(check_any_collisions(poses, base_poly).item()) is False


def test_check_collision_for_index() -> None:
    base_poly = get_tree_polygon()
    poses = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    assert bool(check_collision_for_index(poses, base_poly, jnp.array(0)).item()) is True
    assert bool(check_collision_for_index(poses, base_poly, jnp.array(2)).item()) is False
