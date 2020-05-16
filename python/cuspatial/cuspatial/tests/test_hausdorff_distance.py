# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.tests.utils import assert_eq

import cuspatial


def _test_hausdorff_from_list_of_spaces(spaces):
    return cuspatial.directed_hausdorff_distance(
        [x for space in spaces for (x, y) in space],
        [y for space in spaces for (x, y) in space],
        [len(space) for space in spaces],
    )


def test_empty():
    actual = _test_hausdorff_from_list_of_spaces([])

    expected = cudf.DataFrame([])

    assert_eq(expected, actual)


def test_zeros():
    actual = _test_hausdorff_from_list_of_spaces([[(0, 0)]])

    expected = cudf.DataFrame([0.0])

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


def test_large():
    actual = _test_hausdorff_from_list_of_spaces(
        [[(0.0, 0.0), (0.0, 1.0)], [(-1.0, 0.0), (-1.0, 1.0)]]
    )

    expected = cudf.DataFrame({0: [0.0, 1.0], 1: [1.0, 0.0]})

    assert_eq(expected, actual)


def test_count_one():
    actual = _test_hausdorff_from_list_of_spaces([[(0.0, 0.0)], [(0.0, 1.0)]])

    expected = cudf.DataFrame({0: [0.0, 1.0], 1: [1.0, 0.0]})

    assert_eq(expected, actual)


def test_count_two():
    actual = _test_hausdorff_from_list_of_spaces(
        [[(0.0, 0.0), (0.0, -1.0)], [(1.0, 1.0), (0.0, -1.0)]]
    )

    expected = cudf.DataFrame(
        {0: [0.0, 1.4142135623730951], 1: [1.0, 0.0000000000000000]}
    )

    assert_eq(expected, actual)


def test_values():
    actual = _test_hausdorff_from_list_of_spaces(
        [
            [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (1.0, 7.0)],
            [(3.0, 0.0), (5.0, 2.0), (6.0, 3.0), (5.0, 6.0)],
            [(4.0, 1.0), (7.0, 3.0), (4.0, 6.0)],
        ]
    )

    expected = cudf.DataFrame(
        {
            0: [0.000000, 3.605551, 4.472136],
            1: [4.123106, 0.000000, 1.414214],
            2: [4.000000, 1.414214, 0.000000],
        }
    )

    assert_eq(expected, actual)
