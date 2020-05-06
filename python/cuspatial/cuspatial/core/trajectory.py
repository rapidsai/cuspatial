# Copyright (c) 2019, NVIDIA CORPORATION.

import warnings

from cudf.core import DataFrame, Series
from cudf.core.column import as_column

from cuspatial._lib.trajectory import (
    derive_trajectories as cpp_derive_trajectories,
    trajectory_bounding_boxes as cpp_trajectory_bounding_boxes,
    trajectory_distances_and_speeds as cpp_trajectory_distances_and_speeds,
)

warnings.warn("Duplicates cuDF functionality", DeprecationWarning)


def subset_trajectory_id(trajectory_ids, x_coords, y_coords, point_ids, timestamps):
    """
    Deprecated
    """
    return DataFrame({
        'x': as_column(x_coords),
        'y': as_column(y_coords),
        'ids': as_column(point_ids),
        'timestamp': as_column(timestamps)
    })[trajectory_ids].reset_index(drop=True)


def derive(object_ids, x_coords, y_coords, timestamps):
    """ Derive trajectories from object ids, points, and timestamps.

    Parameters
    ----------
    {params}

    Returns
    -------
    result    : tuple (DataFrame, offsets of of discovered trajectories)
    DataFrame : object_id, x, y, and timestamps, sorted by (object_id, timestamp)
                for calling spatial_bounds and distance_and_speed

    Examples
    --------
    >>> objects, traj_offsets = trajectory.derive(
    >>>    cudf.Series([0, 0, 1, 1]),   # object_id
    >>>    cudf.Series([0, 1, 2, 3]),   # x
    >>>    cudf.Series([0, 0, 1, 1]),   # y
    >>>    cudf.Series([0, 10, 0, 10])) # timestamp
    >>> print(traj_offsets)
        0  0
        1  2
    >>> print(objects)
           object_id       x       y  timestamp
        0          0       1       0          0
        1          0       0       0         10
        2          1       3       1          0
        3          1       2       1         10
    """
    objects, traj_offsets = cpp_derive_trajectories(
        as_column(object_ids),
        as_column(x_coords),
        as_column(y_coords),
        as_column(timestamps)
    )
    return DataFrame._from_table(objects), Series(data=traj_offsets)

def spatial_bounds(num_trajectories, object_ids, x_coords, y_coords):
    """ Compute the bounding boxes of sets of trajectories.

    Parameters
    ----------
    {params}
    result    : DataFrame of x1, y1, x2, y2 as minimum bounding boxes
                (in kilometers) for each trajectory
                   

    Examples
    --------
    >>> objects, traj_offsets = trajectory.derive(
    >>>    cudf.Series([0, 0, 1, 1]),   # object_id
    >>>    cudf.Series([0, 1, 2, 3]),   # x
    >>>    cudf.Series([0, 0, 1, 1]),   # y
    >>>    cudf.Series([0, 10, 0, 10])) # timestamp
    >>> traj_bounding_boxes = trajectory.spatial_bounds(
    >>>     len(traj_offsets),
    >>>     objects['object_id'],
    >>>     objects['x'],
    >>>     objects['y'])
    >>> print(traj_bounding_boxes)
        x1   y1   x2   y2
    0  0.0  0.0  2.0  2.0
    1  1.0  1.0  3.0  3.0
    """
    return DataFrame._from_table(cpp_trajectory_bounding_boxes(
        num_trajectories,
        as_column(object_ids),
        as_column(x_coords),
        as_column(y_coords)
    ))


def distance_and_speed(num_trajectories, object_ids, x_coords, y_coords, timestamps):
    """ Compute the distance traveled and speed of sets of trajectories

    Parameters
    ----------
    {params}

    Returns
    -------
    result : DataFrame
        meters - travelled distance of trajectory
        speed - speed in m/sec of trajectory

    Examples
    --------
    Compute the distance and speed of derived trajectories
    >>> objects, traj_offsets = trajectory.derive(...)
    >>> dists_and_speeds = trajectory.distance_and_speed(len(traj_offsets)
    >>>                                                  objects['object_id'],
    >>>                                                  objects['x'],
    >>>                                                  objects['y'],
    >>>                                                  objects['timestamp'])
    >>> print(dists_and_speeds)
                       distance          speed
        trajectory_id                         
        0                1000.0  100000.000000
        1                1000.0  111111.109375
    """
    df = DataFrame._from_table(cpp_trajectory_distances_and_speeds(
        num_trajectories,
        as_column(object_ids),
        as_column(x_coords),
        as_column(y_coords),
        as_column(timestamps)
    ))
    df.index.name = "trajectory_id"
    return df
    result = cpp_trajectory_distance_and_speed(
        x_coords, y_coords, timestamps, length, position
    )
    df = cudf.DataFrame({"meters": result[0], "speed": result[1]})
    df.index.name = "trajectory_id"
    return df
