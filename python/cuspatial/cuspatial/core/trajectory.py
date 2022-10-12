# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import numpy as np

from cudf import DataFrame, Series
from cudf.core.column import as_column

from cuspatial._lib.trajectory import (
    derive_trajectories as cpp_derive_trajectories,
    trajectory_bounding_boxes as cpp_trajectory_bounding_boxes,
    trajectory_distances_and_speeds as cpp_trajectory_distances_and_speeds,
)
from cuspatial.utils.column_utils import (
    normalize_point_columns,
    normalize_timestamp_column,
)


def derive_trajectories(object_ids, xs, ys, timestamps):
    """
    Derive trajectories from object ids, points, and timestamps.

    Parameters
    ----------
    object_ids
        column of object (e.g., vehicle) ids
    xs
        column of x-coordinates (in kilometers)
    ys
        column of y-coordinates (in kilometers)
    timestamps
        column of timestamps in any resolution

    Returns
    -------
    result : tuple (objects, traj_offsets)
        objects : cudf.DataFrame
            object_ids, xs, ys, and timestamps sorted by
            ``(object_id, timestamp)``, used by ``trajectory_bounding_boxes``
            and ``trajectory_distances_and_speeds``
        traj_offsets : cudf.Series
            offsets of discovered trajectories

    Examples
    --------
    Compute sorted objects and discovered trajectories

    >>> objects, traj_offsets = cuspatial.derive_trajectories(
            [0, 1, 0, 1],  # object_id
            [0, 0, 1, 1],  # x
            [0, 0, 1, 1],  # y
            [0, 10000, 0, 10000] # timestamp
        )
    >>> print(traj_offsets)
        0  0
        1  2
    >>> print(objects)
        object_id    x    y           timestamp
        0          0  0.0  0.0 1970-01-01 00:00:00
        1          0  1.0  1.0 1970-01-01 00:00:10
        2          1  0.0  0.0 1970-01-01 00:00:00
        3          1  1.0  1.0 1970-01-01 00:00:10
    """

    object_ids = as_column(object_ids, dtype=np.int32)
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    timestamps = normalize_timestamp_column(as_column(timestamps))
    objects, traj_offsets = cpp_derive_trajectories(
        object_ids, xs, ys, timestamps
    )
    return DataFrame._from_data(*objects), Series(data=traj_offsets)


def trajectory_bounding_boxes(num_trajectories, object_ids, xs, ys):
    """Compute the bounding boxes of sets of trajectories.

    Parameters
    ----------
    num_trajectories
        number of trajectories (unique object ids)
    object_ids
        column of object (e.g., vehicle) ids
    xs
        column of x-coordinates (in kilometers)
    ys
        column of y-coordinates (in kilometers)

    Returns
    -------
    result : cudf.DataFrame
        minimum bounding boxes (in kilometers) for each trajectory

        x_min : cudf.Series
            the minimum x-coordinate of each bounding box
        y_min : cudf.Series
            the minimum y-coordinate of each bounding box
        x_max : cudf.Series
            the maximum x-coordinate of each bounding box
        y_max : cudf.Series
            the maximum y-coordinate of each bounding box

    Examples
    --------
    Compute the minimum bounding boxes of derived trajectories

    >>> objects, traj_offsets = cuspatial.derive_trajectories(
            [0, 0, 1, 1],  # object_id
            [0, 1, 2, 3],  # x
            [0, 0, 1, 1],  # y
            [0, 10, 0, 10] # timestamp
        )
    >>> traj_bounding_boxes = cuspatial.trajectory_bounding_boxes(
            len(traj_offsets),
            objects['object_id'],
            objects['x'],
            objects['y']
        )
    >>> print(traj_bounding_boxes)
        x_min   y_min   x_max   y_max
    0     0.0     0.0     2.0     2.0
    1     1.0     1.0     3.0     3.0
    """

    object_ids = as_column(object_ids, dtype=np.int32)
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    return DataFrame._from_data(
        *cpp_trajectory_bounding_boxes(num_trajectories, object_ids, xs, ys)
    )


def trajectory_distances_and_speeds(
    num_trajectories, object_ids, xs, ys, timestamps
):
    """
    Compute the distance traveled and speed of sets of trajectories

    Parameters
    ----------
    num_trajectories
        number of trajectories (unique object ids)
    object_ids
        column of object (e.g., vehicle) ids
    xs
        column of x-coordinates (in kilometers)
    ys
        column of y-coordinates (in kilometers)
    timestamps
        column of timestamps in any resolution

    Returns
    -------
    result : cudf.DataFrame
        meters : cudf.Series
            trajectory distance (in kilometers)
        speed  : cudf.Series
            trajectory speed (in meters/second)

    Examples
    --------
    Compute the distances and speeds of derived trajectories

    >>> objects, traj_offsets = cuspatial.derive_trajectories(...)
    >>> dists_and_speeds = cuspatial.trajectory_distances_and_speeds(
            len(traj_offsets)
            objects['object_id'],
            objects['x'],
            objects['y'],
            objects['timestamp']
        )
    >>> print(dists_and_speeds)
                          distance       speed
        trajectory_id
        0              1414.213562  141.421356
        1              1414.213562  141.421356
    """

    object_ids = as_column(object_ids, dtype=np.int32)
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    timestamps = normalize_timestamp_column(as_column(timestamps))
    df = DataFrame._from_data(
        *cpp_trajectory_distances_and_speeds(
            num_trajectories, object_ids, xs, ys, timestamps
        )
    )
    df.index.name = "trajectory_id"
    return df
