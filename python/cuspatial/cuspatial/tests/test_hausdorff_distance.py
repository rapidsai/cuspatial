# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.tests.utils import assert_eq

import cuspatial


def test_empty():
    result = cuspatial.directed_hausdorff_distance(
        cudf.Series(), cudf.Series(), cudf.Series()
    )
    assert_eq(cudf.DataFrame([]), result)


def test_zeros():
    distance = cuspatial.directed_hausdorff_distance(
        cudf.Series([0.0]), cudf.Series([0.0]), cudf.Series([1])
    )
    assert_eq(distance, cudf.DataFrame([0.0]))


def test_empty_x():
    with pytest.raises(RuntimeError):
        cuspatial.directed_hausdorff_distance(
            cudf.Series(), cudf.Series([0]), cudf.Series([0])
        )


def test_empty_y():
    with pytest.raises(RuntimeError):
        cuspatial.directed_hausdorff_distance(
            cudf.Series([0]), cudf.Series(), cudf.Series([0])
        )


def test_large():
    xs = [0.0, 0.0, -1.0, -1.0]
    ys = [0.0, 1.0, 0.0, 1.0]
    space_offsets = [2, 4]

    distance = cuspatial.directed_hausdorff_distance(
        cudf.Series(xs), cudf.Series(ys), cudf.Series(space_offsets)
    )

    assert_eq(distance, cudf.DataFrame({0: [0.0, 1.0], 1: [1.0, 0.0]}))


def test_count_one():
    distance = cuspatial.directed_hausdorff_distance(
        cudf.Series([0.0, 0.0]), cudf.Series([0.0, 1.0]), cudf.Series([1, 1])
    )
    assert_eq(distance, cudf.DataFrame({0: [0.0, 1.0], 1: [1.0, 0.0]}))


def test_count_two():
    distance = cuspatial.directed_hausdorff_distance(
        cudf.Series([0.0, 0.0, 1.0, 0.0]),
        cudf.Series([0.0, -1.0, 1.0, -1.0]),
        cudf.Series([2, 2]),
    )
    assert_eq(
        distance, cudf.DataFrame({0: [0, 1.4142135623730951], 1: [1, 0.0]})
    )


def test_values():
    ys = [0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0]
    xs = [1.0, 2.0, 3.0, 5.0, 7.0, 0.0, 2.0, 3.0, 6.0, 1.0, 3.0, 6.0]
    space_offsets = [5, 4, 3]

    distance = cuspatial.directed_hausdorff_distance(
        cudf.Series(xs), cudf.Series(ys), cudf.Series(space_offsets)
    )

    assert_eq(
        distance,
        cudf.DataFrame(
            {
                0: [0, 3.605551, 4.472136],
                1: [4.123106, 0.0, 1.414214],
                2: [4.0, 1.414214, 0.0],
            }
        ),
    )
