# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import numpy as np
import pytest
from shapely.geometry import Point

import cudf

import cuspatial


def test_trajectory_bounding_boxes_empty_float32():
    result = cuspatial.trajectory_bounding_boxes(
        0,
        cudf.Series(),
        cuspatial.GeoSeries.from_points_xy(cudf.Series([], dtype="f4")),
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x_min": cudf.Series([], dtype=np.float32),
                "y_min": cudf.Series([], dtype=np.float32),
                "x_max": cudf.Series([], dtype=np.float32),
                "y_max": cudf.Series([], dtype=np.float32),
            }
        ),
    )


def test_trajectory_bounding_boxes_empty_float64():
    result = cuspatial.trajectory_bounding_boxes(
        0,
        cudf.Series(),
        cuspatial.GeoSeries.from_points_xy(cudf.Series([], dtype="f8")),
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x_min": cudf.Series([], dtype=np.float64),
                "y_min": cudf.Series([], dtype=np.float64),
                "x_max": cudf.Series([], dtype=np.float64),
                "y_max": cudf.Series([], dtype=np.float64),
            }
        ),
    )


def test_trajectory_bounding_boxes_zeros():
    result = cuspatial.trajectory_bounding_boxes(
        1, cudf.Series([0]), cuspatial.GeoSeries([Point(0, 0)])
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {"x_min": [0.0], "y_min": [0.0], "x_max": [0.0], "y_max": [0.0]}
        ),
    )


def test_trajectory_bounding_boxes_ones():
    result = cuspatial.trajectory_bounding_boxes(
        1, cudf.Series([1]), cuspatial.GeoSeries([Point(1, 1)])
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {"x_min": [1.0], "y_min": [1.0], "x_max": [1.0], "y_max": [1.0]}
        ),
    )


def test_trajectory_bounding_boxes_zero_to_one():
    result = cuspatial.trajectory_bounding_boxes(
        1, cudf.Series([0, 0]), cuspatial.GeoSeries([Point(0, 0), Point(0, 1)])
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {"x_min": [0.0], "y_min": [0.0], "x_max": [0.0], "y_max": [1.0]}
        ),
    )


def test_trajectory_bounding_boxes_zero_to_one_xy():
    result = cuspatial.trajectory_bounding_boxes(
        1, cudf.Series([0, 0]), cuspatial.GeoSeries([Point(0, 0), Point(1, 1)])
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {"x_min": [0.0], "y_min": [0.0], "x_max": [1.0], "y_max": [1.0]}
        ),
    )


def test_trajectory_bounding_boxes_subsetted():
    result = cuspatial.trajectory_bounding_boxes(
        2,
        cudf.Series([0, 0, 1, 1]),
        cuspatial.GeoSeries(
            [Point(0, 0), Point(1, 1), Point(-1, -1), Point(2, 2)]
        ),
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x_min": [0.0, -1.0],
                "y_min": [0.0, -1.0],
                "x_max": [1.0, 2.0],
                "y_max": [1.0, 2.0],
            }
        ),
    )


def test_trajectory_bounding_boxes_intersected():
    result = cuspatial.trajectory_bounding_boxes(
        2,
        cudf.Series([0, 0, 1, 1]),
        cuspatial.GeoSeries(
            [Point(0, 0), Point(2, 2), Point(1, 1), Point(3, 3)]
        ),
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x_min": [0.0, 1.0],
                "y_min": [0.0, 1.0],
                "x_max": [2.0, 3.0],
                "y_max": [2.0, 3.0],
            }
        ),
    )


def test_trajectory_bounding_boxes_two_and_three():
    result = cuspatial.trajectory_bounding_boxes(
        2,
        cudf.Series([0, 0, 1, 1, 1]),
        cuspatial.GeoSeries(
            [Point(0, 0), Point(2, 2), Point(1, 1), Point(3, 3), Point(2, 2)]
        ),
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x_min": [0.0, 1.0],
                "y_min": [0.0, 1.0],
                "x_max": [2.0, 3.0],
                "y_max": [2.0, 3.0],
            }
        ),
    )


def test_derive_trajectories_zeros():
    objects, traj_offsets = cuspatial.derive_trajectories(
        cudf.Series([0]),  # object_id
        cuspatial.GeoSeries([Point(0, 0)]),
        cudf.Series([0]),  # timestamp
    )
    cudf.testing.assert_series_equal(
        traj_offsets, cudf.Series([0], dtype="int32")
    )
    cudf.testing.assert_frame_equal(
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
    objects, traj_offsets = cuspatial.derive_trajectories(
        cudf.Series([1]),  # object_id
        cuspatial.GeoSeries([Point(1, 1)]),
        cudf.Series([1]),  # timestamp
    )
    cudf.testing.assert_series_equal(
        traj_offsets, cudf.Series([0], dtype="int32")
    )
    cudf.testing.assert_frame_equal(
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
    objects, traj_offsets = cuspatial.derive_trajectories(
        cudf.Series([0, 1]),  # object_id
        cuspatial.GeoSeries([Point(0, 0), Point(1, 1)]),
        cudf.Series([0, 1]),  # timestamp
    )
    cudf.testing.assert_series_equal(
        traj_offsets, cudf.Series([0, 1], dtype="int32")
    )
    cudf.testing.assert_frame_equal(
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
    xs = cudf.Series(np.random.randint(0, 10, 10).astype("f8"))
    ys = cudf.Series(np.random.randint(0, 10, 10).astype("f8"))
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame({"x": xs, "y": ys}).interleave_columns()
    )
    timestamp = cudf.Series(np.random.randint(0, 10, 10))
    objects, traj_offsets = cuspatial.derive_trajectories(
        object_id, points, timestamp
    )

    sorted_idxs = cudf.DataFrame({"id": object_id, "ts": timestamp}).argsort()
    cudf.testing.assert_series_equal(
        traj_offsets, cudf.Series([0, 1, 2, 5, 6, 8, 9], dtype="int32")
    )
    cudf.testing.assert_frame_equal(
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


def test_trajectory_distances_and_speeds_zeros():
    points = cuspatial.GeoSeries([Point(0, 0)])
    objects, traj_offsets = cuspatial.derive_trajectories([0], points, [0])
    trajs = cuspatial.GeoSeries.from_points_xy(
        objects[["x", "y"]].interleave_columns()
    )

    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsets),
        objects["object_id"],
        trajs,
        objects["timestamp"],
    )
    cudf.testing.assert_series_equal(
        result["distance"], cudf.Series([0.0]), check_names=False
    )
    cudf.testing.assert_series_equal(
        result["speed"], cudf.Series([0.0]), check_names=False
    )


def test_trajectory_distances_and_speeds_ones():
    points = cuspatial.GeoSeries([Point(0, 0)])
    objects, traj_offsets = cuspatial.derive_trajectories(
        [1],
        points,
        [1],  # object_id  # xs  # ys  # timestamp
    )
    trajs = cuspatial.GeoSeries.from_points_xy(
        objects[["x", "y"]].interleave_columns()
    )

    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsets),
        objects["object_id"],
        trajs,
        objects["timestamp"],
    )
    cudf.testing.assert_series_equal(
        result["distance"], cudf.Series([0.0]), check_names=False
    )
    cudf.testing.assert_series_equal(
        result["speed"], cudf.Series([0.0]), check_names=False
    )


def test_derive_one_trajectory_one_meter_one_second():
    points = cuspatial.GeoSeries([Point(0, 0), Point(0.001, 0.0)])
    objects, traj_offsets = cuspatial.derive_trajectories(
        [0, 0],  # object_id
        points,
        [0, 1000],  # timestamp
    )
    trajs = cuspatial.GeoSeries.from_points_xy(
        objects[["x", "y"]].interleave_columns()
    )

    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsets),
        objects["object_id"],
        trajs,
        objects["timestamp"],
    )
    cudf.testing.assert_series_equal(
        result["distance"], cudf.Series([1.0]), check_names=False
    )
    cudf.testing.assert_series_equal(
        result["speed"], cudf.Series([1.0]), check_names=False
    )


def test_derive_two_trajectories_one_meter_one_second():
    points = cuspatial.GeoSeries(
        [
            Point(0, 0),
            Point(0.001, 0.0),
            Point(0.0, 0.0),
            Point(0.0, 0.001),
        ]
    )
    objects, traj_offsets = cuspatial.derive_trajectories(
        [0, 0, 1, 1],  # object_id
        points,
        [0, 1000, 0, 1000],  # timestamp
    )
    trajs = cuspatial.GeoSeries.from_points_xy(
        objects[["x", "y"]].interleave_columns()
    )

    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsets),
        objects["object_id"],
        trajs,
        objects["timestamp"],
    )
    cudf.testing.assert_series_equal(
        result["distance"], cudf.Series([1.0, 1.0]), check_names=False
    )
    cudf.testing.assert_series_equal(
        result["speed"], cudf.Series([1.0, 1.0]), check_names=False
    )


def test_trajectory_distances_and_speeds_single_trajectory():
    x = cudf.Series(
        [1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0]
    )  # xs
    y = cudf.Series(
        [0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0]
    )  # ys

    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame({"x": x, "y": y}).interleave_columns()
    )

    objects, traj_offsets = cuspatial.derive_trajectories(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],  # object_id
        points,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # timestamp
    )
    trajs = cuspatial.GeoSeries.from_points_xy(
        objects[["x", "y"]].interleave_columns()
    )
    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsets),
        objects["object_id"],
        trajs,
        objects["timestamp"],
    )
    cudf.testing.assert_series_equal(
        result["distance"],
        cudf.Series([7892.922363, 6812.55908203125, 8485.28125]),
        check_names=False,
    )
    cudf.testing.assert_series_equal(
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
def test_trajectory_distances_and_speeds_timestamp_types(timestamp_type):
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame(
            {
                "x": cudf.Series([0.0, 0.001, 0.0, 0.0]),
                "y": cudf.Series([0.0, 0.0, 0.0, 0.001]),
            }
        ).interleave_columns()
    )
    objects, traj_offsets = cuspatial.derive_trajectories(
        # object_id
        cudf.Series([0, 0, 1, 1]),
        points,
        # timestamp
        cudf.Series([0, timestamp_type[1], 0, timestamp_type[1]]).astype(
            timestamp_type[0]
        ),
    )
    trajs = cuspatial.GeoSeries.from_points_xy(
        objects[["x", "y"]].interleave_columns()
    )
    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsets),
        objects["object_id"],
        trajs,
        objects["timestamp"],
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame({"distance": [1.0, 1.0], "speed": [1.0, 1.0]}),
        check_names=False,
    )
