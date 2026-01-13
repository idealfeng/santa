import unittest

import numpy as np

from santa_packing.constants import XY_LIMIT
from santa_packing.submission_format import fit_xy_in_bounds, format_submission_value, quantize_for_submission


class TestSubmissionFormat(unittest.TestCase):
    def test_format_negative_zero(self) -> None:
        self.assertEqual(format_submission_value(-0.0), format_submission_value(0.0))

    def test_fit_xy_in_bounds(self) -> None:
        poses = np.array([[200.0, 200.0, 0.0], [201.0, 201.0, 0.0]], dtype=float)
        fitted = fit_xy_in_bounds(poses)
        self.assertLessEqual(float(np.max(fitted[:, 0])), XY_LIMIT)
        self.assertGreaterEqual(float(np.min(fitted[:, 0])), -XY_LIMIT)
        self.assertLessEqual(float(np.max(fitted[:, 1])), XY_LIMIT)
        self.assertGreaterEqual(float(np.min(fitted[:, 1])), -XY_LIMIT)

    def test_quantize_for_submission(self) -> None:
        poses = np.array([[1000.0, -1000.0, 361.0], [-1e-12, 1e-12, -0.0]], dtype=float)
        q = quantize_for_submission(poses)
        self.assertTrue(np.all(q[:, 0] <= XY_LIMIT))
        self.assertTrue(np.all(q[:, 0] >= -XY_LIMIT))
        self.assertTrue(np.all(q[:, 1] <= XY_LIMIT))
        self.assertTrue(np.all(q[:, 1] >= -XY_LIMIT))
        self.assertTrue(np.all((0.0 <= q[:, 2]) & (q[:, 2] < 360.0)))
        self.assertEqual(float(q[1, 0]), 0.0)
        self.assertEqual(float(q[1, 1]), 0.0)
        self.assertEqual(float(q[1, 2]), 0.0)
