# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

import cuspatial


def test_subset_id_zeros():
    result = cuspatial.subset_trajectory_id(
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
                "ids": cudf.Series([0]).astype("int32"),
                "timestamp": cudf.Series([0]).astype("datetime64[ms]"),
            }
        ),
    )


def test_subset_id_ones():
    result = cuspatial.subset_trajectory_id(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "x": [1.0],
                "y": [1.0],
                "ids": cudf.Series([1]).astype("int32"),
                "timestamp": cudf.Series([1]).astype("datetime64[ms]"),
            }
        ),
    )


def test_subset_id_random():
    np.random.seed(0)
    result = cuspatial.subset_trajectory_id(
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "x": [7.0, 6, 1, 6, 7, 7, 8],
                "y": [5.0, 9, 4, 3, 0, 3, 5],
                "ids": cudf.Series([2, 3, 3, 3, 3, 7, 0]).astype("int32"),
                "timestamp": cudf.Series([9, 9, 7, 3, 2, 7, 2]).astype(
                    "datetime64[ms]"
                ),
            }
        ),
    )


def test_spatial_bounds_zeros():
    result = cuspatial.spatial_bounds(
        cudf.Series([0]), cudf.Series([0]), cudf.Series([0]), cudf.Series([0])
    )
    assert_eq(
        result,
        cudf.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [0.0], "y2": [0.0]}),
    )


def test_spatial_bounds_ones():
    result = cuspatial.spatial_bounds(
        cudf.Series([1]), cudf.Series([1]), cudf.Series([1]), cudf.Series([1])
    )
    assert_eq(
        result,
        cudf.DataFrame({"x1": [1.0], "y1": [1.0], "x2": [1.0], "y2": [1.0]}),
    )


def test_spatial_bounds_zero_to_one():
    result = cuspatial.spatial_bounds(
        cudf.Series([0, 0]),
        cudf.Series([0, 1]),
        cudf.Series([2]),
        cudf.Series([2]),
    )
    assert_eq(
        result,
        cudf.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [0.0], "y2": [1.0]}),
    )


def test_spatial_bounds_zero_to_one_xy():
    result = cuspatial.spatial_bounds(
        cudf.Series([0, 1]),
        cudf.Series([0, 1]),
        cudf.Series([2]),
        cudf.Series([2]),
    )
    assert_eq(
        result,
        cudf.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [1.0], "y2": [1.0]}),
    )


def test_spatial_bounds_subsetted():
    result = cuspatial.spatial_bounds(
        cudf.Series([0, 1, -1, 2]),
        cudf.Series([0, 1, -1, 2]),
        cudf.Series([2, 2]),
        cudf.Series([2, 4]),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "x1": [0.0, -1.0],
                "y1": [0.0, -1.0],
                "x2": [1.0, 2.0],
                "y2": [1.0, 2.0],
            }
        ),
    )


def test_spatial_bounds_intersected():
    result = cuspatial.spatial_bounds(
        cudf.Series([0, 2, 1, 3]),
        cudf.Series([0, 2, 1, 3]),
        cudf.Series([2, 2]),
        cudf.Series([2, 4]),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "x1": [0.0, 1.0],
                "y1": [0.0, 1.0],
                "x2": [2.0, 3.0],
                "y2": [2.0, 3.0],
            }
        ),
    )


def test_spatial_bounds_two_and_three():
    result = cuspatial.spatial_bounds(
        cudf.Series([0, 2, 1, 3, 2]),
        cudf.Series([0, 2, 1, 3, 2]),
        cudf.Series([2, 3]),
        cudf.Series([2, 5]),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "x1": [0.0, 1.0],
                "y1": [0.0, 1.0],
                "x2": [2.0, 3.0],
                "y2": [2.0, 3.0],
            }
        ),
    )


def test_derive_trajectories_zeros():
    num_trajectories = cuspatial.derive(
        cudf.Series([0]), cudf.Series([0]), cudf.Series([0]), cudf.Series([0])
    )
    assert num_trajectories[0] == 1
    assert_eq(
        num_trajectories[1],
        cudf.DataFrame(
            {
                "trajectory_id": cudf.Series([0]).astype("int32"),
                "length": cudf.Series([1]).astype("int32"),
                "position": cudf.Series([1]).astype("int32"),
            }
        ),
    )


def test_derive_trajectories_ones():
    num_trajectories = cuspatial.derive(
        cudf.Series([1]), cudf.Series([1]), cudf.Series([1]), cudf.Series([1])
    )
    assert num_trajectories[0] == 1
    assert_eq(
        num_trajectories[1],
        cudf.DataFrame(
            {
                "trajectory_id": cudf.Series([1]).astype("int32"),
                "length": cudf.Series([1]).astype("int32"),
                "position": cudf.Series([1]).astype("int32"),
            }
        ),
    )


def test_derive_trajectories_two():
    num_trajectories = cuspatial.derive(
        cudf.Series([0, 1]),
        cudf.Series([0, 1]),
        cudf.Series([0, 1]),
        cudf.Series([0, 1]),
    )
    assert num_trajectories[0] == 2
    assert_eq(
        num_trajectories[1],
        cudf.DataFrame(
            {
                "trajectory_id": cudf.Series([0, 1]).astype("int32"),
                "length": cudf.Series([1, 1]).astype("int32"),
                "position": cudf.Series([1, 2]).astype("int32"),
            }
        ),
    )


def test_derive_trajectories_many():
    np.random.seed(0)
    num_trajectories = cuspatial.derive(
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
    )
    assert num_trajectories[0] == 6
    assert_eq(
        num_trajectories[1],
        cudf.DataFrame(
            {
                "trajectory_id": cudf.Series([0, 3, 4, 5, 8, 9]).astype(
                    "int32"
                ),
                "length": cudf.Series([2, 2, 1, 2, 1, 2]).astype("int32"),
                "position": cudf.Series([2, 4, 5, 7, 8, 10]).astype("int32"),
            }
        ),
    )


def test_distance_and_speed_zeros():
    result = cuspatial.distance_and_speed(
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
    )
    assert_eq(result["meters"], cudf.Series([-2.0]), check_names=False)
    assert_eq(result["speed"], cudf.Series([-2.0]), check_names=False)


def test_distance_and_speed_ones():
    result = cuspatial.distance_and_speed(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
    )
    assert_eq(result["meters"], cudf.Series([-2.0]), check_names=False)
    assert_eq(result["speed"], cudf.Series([-2.0]), check_names=False)


def test_one_one_meter_one_second():
    result = cuspatial.distance_and_speed(
        cudf.Series([0.0, 0.001]),
        cudf.Series([0.0, 0.0]),
        cudf.Series([0, 1000]),
        cudf.Series([2]),
        cudf.Series([2]),
    )
    assert_eq(result["meters"], cudf.Series([1.0]), check_names=False)
    assert_eq(result["speed"], cudf.Series([1.0]), check_names=False)


def test_two_trajectories_one_meter_one_second():
    result = cuspatial.distance_and_speed(
        cudf.Series([0.0, 0.001, 0.0, 0.0]),
        cudf.Series([0.0, 0.0, 0.0, 0.001]),
        cudf.Series([0, 1000, 0, 1000]),
        cudf.Series([2, 2]),
        cudf.Series([2, 4]),
    )
    assert_eq(result["meters"], cudf.Series([1.0, 1.0]), check_names=False)
    assert_eq(result["speed"], cudf.Series([1.0, 1.0]), check_names=False)


def test_distance_and_speed_single_trajectory():
    result = cuspatial.distance_and_speed(
        cudf.Series(
            [1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0]
        ),
        cudf.Series(
            [0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0]
        ),
        cudf.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        cudf.Series([5, 4, 3]),
        cudf.Series([5, 9, 12]),
    )
    assert_eq(
        result["meters"],
        cudf.Series([7892.922363, 6812.55908203125, 8485.28125]),
        check_names=False,
    )
    assert_eq(
        result["speed"],
        cudf.Series([1973230.625, 2270853.0, 4242640.5]),
        check_names=False,
    )  # fast!


#########################
# Measure that distance and speed are calculatd
# correctly using each of the four cudf datetime
# resolutions.
#
# Compute the distance and speed of two trajectories,
# each over 0.001 km in 1 second.
# If datetime type conversion wasn't supported, speed
# would be different for each test.
#########################
@pytest.mark.parametrize(
    "timestamp_type",
    [
        ("datetime64[ns]", 1000000000),
        ("datetime64[us]", 1000000),
        ("datetime64[ms]", 1000),
        ("datetime64[s]", 1),
    ],
)
def test_distance_and_speed_timestamp_types(timestamp_type):
    result = cuspatial.distance_and_speed(
        cudf.Series([0.0, 0.001, 0.0, 0.0]),  # 1 meter in x
        cudf.Series([0.0, 0.0, 0.0, 0.001]),  # 1 meter in y
        cudf.Series([0, timestamp_type[1], 0, timestamp_type[1]]).astype(
            timestamp_type[0]
        ),
        cudf.Series([2, 2]),
        cudf.Series([2, 4]),
    )
    assert_eq(
        result,
        cudf.DataFrame({"meters": [1.0, 1.0], "speed": [1.0, 1.0]}),
        check_names=False,
    )
