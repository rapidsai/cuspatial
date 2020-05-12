# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

import cuspatial


def test_spatial_bounds_empty_float32():
    result = cuspatial.spatial_bounds(
        0,
        cudf.Series(),
        cudf.Series([], dtype=np.float32),
        cudf.Series([], dtype=np.float32),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "x1": cudf.Series([], dtype=np.float32),
                "y1": cudf.Series([], dtype=np.float32),
                "x2": cudf.Series([], dtype=np.float32),
                "y2": cudf.Series([], dtype=np.float32),
            }
        ),
    )


def test_spatial_bounds_empty_float64():
    result = cuspatial.spatial_bounds(
        0,
        cudf.Series(),
        cudf.Series([], dtype=np.float64),
        cudf.Series([], dtype=np.float64),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "x1": cudf.Series([], dtype=np.float64),
                "y1": cudf.Series([], dtype=np.float64),
                "x2": cudf.Series([], dtype=np.float64),
                "y2": cudf.Series([], dtype=np.float64),
            }
        ),
    )


def test_spatial_bounds_zeros():
    result = cuspatial.spatial_bounds(
        1, cudf.Series([0]), cudf.Series([0]), cudf.Series([0])
    )
    assert_eq(
        result,
        cudf.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [0.0], "y2": [0.0]}),
    )


def test_spatial_bounds_ones():
    result = cuspatial.spatial_bounds(
        1, cudf.Series([1]), cudf.Series([1]), cudf.Series([1])
    )
    assert_eq(
        result,
        cudf.DataFrame({"x1": [1.0], "y1": [1.0], "x2": [1.0], "y2": [1.0]}),
    )


def test_spatial_bounds_zero_to_one():
    result = cuspatial.spatial_bounds(
        1, cudf.Series([0, 0]), cudf.Series([0, 0]), cudf.Series([0, 1]),
    )
    assert_eq(
        result,
        cudf.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [0.0], "y2": [1.0]}),
    )


def test_spatial_bounds_zero_to_one_xy():
    result = cuspatial.spatial_bounds(
        1, cudf.Series([0, 0]), cudf.Series([0, 1]), cudf.Series([0, 1]),
    )
    assert_eq(
        result,
        cudf.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [1.0], "y2": [1.0]}),
    )


def test_spatial_bounds_subsetted():
    result = cuspatial.spatial_bounds(
        2,
        cudf.Series([0, 0, 1, 1]),
        cudf.Series([0, 1, -1, 2]),
        cudf.Series([0, 1, -1, 2]),
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
        2,
        cudf.Series([0, 0, 1, 1]),
        cudf.Series([0, 2, 1, 3]),
        cudf.Series([0, 2, 1, 3]),
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
        2,
        cudf.Series([0, 0, 1, 1, 1]),
        cudf.Series([0, 2, 1, 3, 2]),
        cudf.Series([0, 2, 1, 3, 2]),
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
    objects, traj_offsets = cuspatial.derive(
        cudf.Series([0]),  # object_id
        cudf.Series([0]),  # x
        cudf.Series([0]),  # y
        cudf.Series([0]),  # timestamp
    )
    assert_eq(traj_offsets, cudf.Series([0], dtype="int32"))
    assert_eq(
        objects,
        cudf.DataFrame(
            {
                "object_id": cudf.Series([0], dtype="int32"),
                "x": cudf.Series([0], dtype="float64"),
                "y": cudf.Series([0], dtype="float64"),
                "timestamp": cudf.Series([0], dtype="datetime64[ms]"),
            }
        ),
    )


def test_derive_trajectories_ones():
    objects, traj_offsets = cuspatial.derive(
        cudf.Series([1]),  # object_id
        cudf.Series([1]),  # x
        cudf.Series([1]),  # y
        cudf.Series([1]),  # timestamp
    )
    assert_eq(traj_offsets, cudf.Series([0], dtype="int32"))
    assert_eq(
        objects,
        cudf.DataFrame(
            {
                "object_id": cudf.Series([1], dtype="int32"),
                "x": cudf.Series([1], dtype="float64"),
                "y": cudf.Series([1], dtype="float64"),
                "timestamp": cudf.Series([1], dtype="datetime64[ms]"),
            }
        ),
    )


def test_derive_trajectories_two():
    objects, traj_offsets = cuspatial.derive(
        cudf.Series([0, 1]),  # object_id
        cudf.Series([0, 1]),  # x
        cudf.Series([0, 1]),  # y
        cudf.Series([0, 1]),  # timestamp
    )
    assert_eq(traj_offsets, cudf.Series([0, 1], dtype="int32"))
    assert_eq(
        objects,
        cudf.DataFrame(
            {
                "object_id": cudf.Series([0, 1], dtype="int32"),
                "x": cudf.Series([0, 1], dtype="float64"),
                "y": cudf.Series([0, 1], dtype="float64"),
                "timestamp": cudf.Series([0, 1], dtype="datetime64[ms]"),
            }
        ),
    )


def test_derive_trajectories_many():
    np.random.seed(0)
    object_id = cudf.Series(np.random.randint(0, 10, 10), dtype="int32")
    xs = cudf.Series(np.random.randint(0, 10, 10))
    ys = cudf.Series(np.random.randint(0, 10, 10))
    timestamp = cudf.Series(np.random.randint(0, 10, 10))
    objects, traj_offsets = cuspatial.derive(object_id, xs, ys, timestamp)

    sorted_idxs = cudf.DataFrame({"id": object_id, "ts": timestamp}).argsort()
    assert_eq(traj_offsets, cudf.Series([0, 1, 2, 5, 6, 8, 9], dtype="int32"))
    print(objects)
    assert_eq(
        objects,
        cudf.DataFrame(
            {
                "object_id": object_id.sort_values().reset_index(drop=True),
                "x": xs.take(sorted_idxs)
                .reset_index(drop=True)
                .astype("float64"),
                "y": ys.take(sorted_idxs)
                .reset_index(drop=True)
                .astype("float64"),
                "timestamp": timestamp.take(sorted_idxs)
                .reset_index(drop=True)
                .astype("datetime64[ms]"),
            },
            index=cudf.core.index.RangeIndex(0, 10),
        ),
    )


def test_distance_and_speed_zeros():
    objects, traj_offsets = cuspatial.derive(
        [0], [0], [0], [0],  # object_id  # xs  # ys  # timestamp
    )
    result = cuspatial.distance_and_speed(
        len(traj_offsets),
        objects["object_id"],
        objects["x"],
        objects["y"],
        objects["timestamp"],
    )
    assert_eq(result["distance"], cudf.Series([0.0]), check_names=False)
    assert_eq(result["speed"], cudf.Series([0.0]), check_names=False)


def test_distance_and_speed_ones():
    objects, traj_offsets = cuspatial.derive(
        [1], [1], [1], [1],  # object_id  # xs  # ys  # timestamp
    )
    result = cuspatial.distance_and_speed(
        len(traj_offsets),
        objects["object_id"],
        objects["x"],
        objects["y"],
        objects["timestamp"],
    )
    assert_eq(result["distance"], cudf.Series([0.0]), check_names=False)
    assert_eq(result["speed"], cudf.Series([0.0]), check_names=False)


def test_one_one_meter_one_second():
    objects, traj_offsets = cuspatial.derive(
        [0, 0],  # object_id
        [0.0, 0.001],  # xs
        [0.0, 0.0],  # ys
        [0, 1000],  # timestamp
    )
    result = cuspatial.distance_and_speed(
        len(traj_offsets),
        objects["object_id"],
        objects["x"],
        objects["y"],
        objects["timestamp"],
    )
    assert_eq(result["distance"], cudf.Series([1.0]), check_names=False)
    assert_eq(result["speed"], cudf.Series([1.0]), check_names=False)


def test_two_trajectories_one_meter_one_second():
    objects, traj_offsets = cuspatial.derive(
        [0, 0, 1, 1],  # object_id
        [0.0, 0.001, 0.0, 0.0],  # xs
        [0.0, 0.0, 0.0, 0.001],  # ys
        [0, 1000, 0, 1000],  # timestamp
    )
    result = cuspatial.distance_and_speed(
        len(traj_offsets),
        objects["object_id"],
        objects["x"],
        objects["y"],
        objects["timestamp"],
    )
    assert_eq(result["distance"], cudf.Series([1.0, 1.0]), check_names=False)
    assert_eq(result["speed"], cudf.Series([1.0, 1.0]), check_names=False)


def test_distance_and_speed_single_trajectory():
    objects, traj_offsets = cuspatial.derive(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],  # object_id
        [1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0],  # xs
        [0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0],  # ys
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # timestamp
    )
    result = cuspatial.distance_and_speed(
        len(traj_offsets),
        objects["object_id"],
        objects["x"],
        objects["y"],
        objects["timestamp"],
    )
    assert_eq(
        result["distance"],
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
    objects, traj_offsets = cuspatial.derive(
        # object_id
        cudf.Series([0, 0, 1, 1]),
        # xs
        cudf.Series([0.0, 0.001, 0.0, 0.0]),  # 1 meter in x
        # ys
        cudf.Series([0.0, 0.0, 0.0, 0.001]),  # 1 meter in y
        # timestamp
        cudf.Series([0, timestamp_type[1], 0, timestamp_type[1]]).astype(
            timestamp_type[0]
        ),
    )
    result = cuspatial.distance_and_speed(
        len(traj_offsets),
        objects["object_id"],
        objects["x"],
        objects["y"],
        objects["timestamp"],
    )
    assert_eq(
        result,
        cudf.DataFrame({"distance": [1.0, 1.0], "speed": [1.0, 1.0]}),
        check_names=False,
    )
