import numpy as np
import pytest

try:
    import jax.numpy as jnp

    from santa_packing.geometry import polygon_bbox, rotate_point, transform_polygon
except Exception:  # pragma: no cover
    pytest.skip("JAX not available", allow_module_level=True)


def test_rotate_point_90_deg() -> None:
    p = jnp.array([1.0, 0.0])
    out = rotate_point(p, jnp.pi / 2.0)
    np.testing.assert_allclose(np.array(out), np.array([0.0, 1.0]), atol=1e-6, rtol=0.0)


def test_transform_polygon_rotation_and_translation() -> None:
    poly = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    pose = jnp.array([2.0, 3.0, 90.0])
    out = transform_polygon(poly, pose)
    expected = np.array([[2.0, 4.0], [1.0, 3.0]])
    np.testing.assert_allclose(np.array(out), expected, atol=1e-6, rtol=0.0)


def test_polygon_bbox() -> None:
    poly = jnp.array([[2.0, -1.0], [0.5, 4.0], [3.0, 1.0]])
    bbox = polygon_bbox(poly)
    np.testing.assert_allclose(np.array(bbox), np.array([0.5, -1.0, 3.0, 4.0]), atol=1e-12, rtol=0.0)
