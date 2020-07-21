# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

import cuspatial


def _test_polygon_distance_from_list_of_spaces(spaces):
    lengths = [len(space) for space in spaces]
    offsets = np.cumsum([0, *lengths])[:-1]
    return cuspatial.directed_polygon_distance(
        [x for space in spaces for (x, y) in space],
        [y for space in spaces for (x, y) in space],
        offsets,
    )


def test_empty():
    actual = _test_polygon_distance_from_list_of_spaces([])
    expected = cudf.DataFrame([])
    assert_eq(expected, actual)


def test_empty_x():
    with pytest.raises(RuntimeError):
        cuspatial.directed_hausdorff_distance(
            [], [0.0], [0],
        )


def test_empty_y():
    with pytest.raises(RuntimeError):
        cuspatial.directed_hausdorff_distance(
            [0.0], [], [0],
        )


def test_solo_point():
    actual = _test_polygon_distance_from_list_of_spaces([[(0, 0)]])
    expected = cudf.DataFrame([0.0])
    assert_eq(expected, actual)


def test_solo_edge():
    actual = _test_polygon_distance_from_list_of_spaces([[(1, 2), (2, 1)]])
    expected = cudf.DataFrame([0.0])
    assert_eq(expected, actual)


def test_solo_edge_zero_length():
    actual = _test_polygon_distance_from_list_of_spaces([[(0, 0), (0, 0)]])
    expected = cudf.DataFrame([0.0])
    assert_eq(expected, actual)


def test_solo_triangle():
    actual = _test_polygon_distance_from_list_of_spaces(
        [[(-1, 0), (0, 1), (1, 0)]]
    )

    expected = cudf.DataFrame([0.0])
    assert_eq(expected, actual)


def test_two_triangles_edge_to_point():
    actual = _test_polygon_distance_from_list_of_spaces(
        [[(-1, 0), (0, 1), (1, 0)]]
    )

    expected = cudf.DataFrame([0.0])
    assert_eq(expected, actual)


def test_two_triangles_point_to_point():
    actual = _test_polygon_distance_from_list_of_spaces(
        [[(-1, 3), (0, 1), (1, 3)], [(-1, -3), (0, -1), (1, -3)]]
    )

    expected = cudf.DataFrame({0: [0.0, 2.0], 1: [2.0, 0.0]})
    assert_eq(expected, actual)


def test_simultaneously_point_edge_triangle_quad():
    actual = cuspatial.directed_polygon_distance(
        [4, 2, 5, 1, 2, 4, 0, 1, 5, 6],
        [3, 1, 2, 2, 4, 5, 1, 5, 6, 0],
        [0, 1, 3, 6],
    )

    expected = cudf.DataFrame(
        {
            0: [0.000000, 1.414214, 2.000000, 3.162278],
            1: [1.264911, 0.000000, 1.414214, 2.000000],
            2: [1.414214, 1.414214, 0.000000, 1.414214],
            3: [1.479591, 0.328798, 0.727607, 0.000000],
        }
    )

    assert_eq(expected, actual)
