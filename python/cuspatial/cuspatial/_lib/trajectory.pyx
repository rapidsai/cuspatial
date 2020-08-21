# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cuspatial._lib.cpp.trajectory cimport (
    derive_trajectories as cpp_derive_trajectories,
    trajectory_bounding_boxes as cpp_trajectory_bounding_boxes,
    trajectory_distances_and_speeds as cpp_trajectory_distances_and_speeds,
)

from cuspatial._lib.move cimport move

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

cpdef derive_trajectories(Column object_id, Column x,
                          Column y, Column timestamp):
    cdef column_view c_id = object_id.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef column_view c_ts = timestamp.view()
    cdef pair[unique_ptr[table], unique_ptr[column]] result
    with nogil:
        result = move(cpp_derive_trajectories(c_id, c_x, c_y, c_ts))
    table = Table.from_unique_ptr(
        move(result.first),
        column_names=["object_id", "x", "y", "timestamp"]
    )
    return table, Column.from_unique_ptr(move(result.second))


cpdef trajectory_bounding_boxes(size_type num_trajectories,
                                Column object_id, Column x, Column y):
    cdef column_view c_id = object_id.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_trajectory_bounding_boxes(
            num_trajectories, c_id, c_x, c_y
        ))
    return Table.from_unique_ptr(
        move(result),
        column_names=["x_min", "y_min", "x_max", "y_max"]
    )


cpdef trajectory_distances_and_speeds(size_type num_trajectories,
                                      Column object_id, Column x,
                                      Column y, Column timestamp):
    cdef column_view c_id = object_id.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef column_view c_ts = timestamp.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_trajectory_distances_and_speeds(
            num_trajectories, c_id, c_x, c_y, c_ts
        ))
    return Table.from_unique_ptr(
        move(result),
        column_names=["distance", "speed"]
    )
