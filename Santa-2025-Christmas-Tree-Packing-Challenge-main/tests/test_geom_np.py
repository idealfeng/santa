import unittest

import numpy as np

from santa_packing.geom_np import transform_polygon


class TestGeomNP(unittest.TestCase):
    def test_transform_polygon_translation(self) -> None:
        pts = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=float)
        pose = np.array([3.5, -4.0, 0.0], dtype=float)
        out = transform_polygon(pts, pose)
        np.testing.assert_allclose(out, pts + np.array([3.5, -4.0]), atol=1e-12, rtol=0.0)

    def test_transform_polygon_rotation_360(self) -> None:
        pts = np.array([[0.25, -1.5], [2.0, 3.0], [-4.0, 0.0]], dtype=float)
        out0 = transform_polygon(pts, np.array([0.0, 0.0, 0.0], dtype=float))
        out360 = transform_polygon(pts, np.array([0.0, 0.0, 360.0], dtype=float))
        np.testing.assert_allclose(out0, out360, atol=1e-12, rtol=0.0)

    def test_transform_polygon_rotation_90(self) -> None:
        pts = np.array([[1.0, 0.0]], dtype=float)
        out = transform_polygon(pts, np.array([0.0, 0.0, 90.0], dtype=float))
        np.testing.assert_allclose(out, np.array([[0.0, 1.0]]), atol=1e-12, rtol=0.0)
