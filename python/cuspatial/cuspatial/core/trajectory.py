# Copyright (c) 2019, NVIDIA CORPORATION.

import warnings

import cudf

from cuspatial._lib.trajectory import (
    cpp_derive_trajectories,
    cpp_subset_trajectory_id,
    cpp_trajectory_distance_and_speed,
    cpp_trajectory_spatial_bounds,
)

warnings.warn("Duplicates cuDF functionality", DeprecationWarning)


def subset_trajectory_id(trajectory_ids, in_x, in_y, point_ids, timestamps):
    """
    Deprecated
    """
    return cpp_subset_trajectory_id(
        trajectory_ids, in_x, in_y, point_ids, timestamps
    )


def spatial_bounds(
    x_coords, y_coords, trajectory_size, trajectory_end_position
):
    """ Compute the bounding boxes of sets of trajectories.

    Parameters
    ----------
    {params}

    Examples
    --------
    >>> result = trajectory.spatial_bounds(
    >>>    cudf.Series([0, 2, 1, 3, 2]),
    >>>    cudf.Series([0, 2, 1, 3, 2]),
    >>>    cudf.Series([2, 3]),
    >>>    cudf.Series([2, 5])
    >>> )
    >>> print(result)
        x1   y1   x2   y2
    0  0.0  0.0  2.0  2.0
    1  1.0  1.0  3.0  3.0
    """
    return cpp_trajectory_spatial_bounds(
        x_coords, y_coords, trajectory_size, trajectory_end_position
    )


def derive(x_coords, y_coords, object_ids, timestamps):
    """ Derive trajectories from points, timestamps, and ids.

    Parameters
    ----------
    {params}

    Returns
    -------
    result_tuple : tuple (number of discovered trajectories,DataFrame)
    DataFrame    : id, length, and positions of trajectories
                   for feeding into compute_distance_and_speed

    Examples
    --------
    >>> num_trajectories, result = trajectory.derive(
    >>>    cudf.Series([0, 1, 2, 3]),
    >>>    cudf.Series([0, 0, 1, 1]),
    >>>    cudf.Series([0, 0, 1, 1]),
    >>>    cudf.Series([0, 10, 0, 10]))
    >>> print(num_trajectories)
        2
    >>> print(result)
           trajectory_id  length  position
        0              0       2         2
        1              1       2         4
    """
    return cpp_derive_trajectories(x_coords, y_coords, object_ids, timestamps)


def distance_and_speed(x_coords, y_coords, timestamps, length, position):
    """ Compute the distance travelled and speed of sets of trajectories

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
    Compute the distance and speed of the above derived trajectories
    >>> result = trajectory.distance_and_speed(x, y, timestamps,
                                               result['length'],
                                               result['position'])
    >>> print(result)
                       meters          speed
        trajectory_id
        0              1000.0  100000.000000
        1              1000.0  111111.109375
    """
    result = cpp_trajectory_distance_and_speed(
        x_coords, y_coords, timestamps, length, position
    )
    df = cudf.DataFrame({"meters": result[0], "speed": result[1]})
    df.index.name = "trajectory_id"
    return df
