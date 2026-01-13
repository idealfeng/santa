import numpy as np

from santa_packing.geom_np import transform_polygon
from santa_packing.scoring import first_overlap_pair, polygons_intersect, polygons_intersect_strict
from santa_packing.tree_data import TREE_POINTS

SQUARE = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    dtype=float,
)


def test_overlap() -> None:
    a = SQUARE
    b = SQUARE + np.array([0.5, 0.5], dtype=float)
    assert polygons_intersect_strict(a, b)
    assert polygons_intersect(a, b)


def test_touch_counts_as_intersection() -> None:
    a = SQUARE
    b = SQUARE + np.array([1.0, 0.0], dtype=float)
    assert not polygons_intersect_strict(a, b)
    assert polygons_intersect(a, b)


def test_separated() -> None:
    a = SQUARE
    b = SQUARE + np.array([2.0, 0.0], dtype=float)
    assert not polygons_intersect_strict(a, b)
    assert not polygons_intersect(a, b)


def test_kaggle_mode_has_no_false_negative_from_scaling() -> None:
    points = np.array(TREE_POINTS, dtype=float)
    poses = np.array(
        [
            [0.0, 0.0, 304.29219185],
            [-0.26471355, 0.76206731, 176.13282888],
        ],
        dtype=float,
    )

    assert first_overlap_pair(points, poses, mode="strict") == (0, 1)
    assert first_overlap_pair(points, poses, mode="kaggle") == (0, 1)

    # Uniformly scaling the *local* polygon about the origin is not a valid way to
    # model "clearance" for concave polygons: it can remove intersections.
    scaled = points * 1.0005
    a = transform_polygon(points, poses[0])
    b = transform_polygon(points, poses[1])
    a_scaled = transform_polygon(scaled, poses[0])
    b_scaled = transform_polygon(scaled, poses[1])
    assert polygons_intersect_strict(a, b)
    assert not polygons_intersect_strict(a_scaled, b_scaled)


def test_strict_does_not_flag_concave_touch() -> None:
    # Regression: some concave touch configurations used to be reported as strict
    # intersections when the optional fast collider was enabled.
    points = np.array(TREE_POINTS, dtype=float)
    poses = np.array(
        [
            [0.64562313, 0.84968406, 203.62937773],
            [0.33742899, 0.32676555, 23.62937773],
        ],
        dtype=float,
    )

    # Touching counts as collision for the conservative predicate.
    a = transform_polygon(points, poses[0])
    b = transform_polygon(points, poses[1])
    assert not polygons_intersect_strict(a, b)
    assert polygons_intersect(a, b)

    # Therefore, strict mode should not report an overlap pair here.
    assert first_overlap_pair(points, poses, mode="strict") is None
    assert first_overlap_pair(points, poses, mode="kaggle") is None
