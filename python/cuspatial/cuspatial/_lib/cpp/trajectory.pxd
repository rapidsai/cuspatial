# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type


cdef extern from "cuspatial/trajectory.hpp" namespace "cuspatial" nogil:

    cdef pair[unique_ptr[table], unique_ptr[column]] derive_trajectories(
        const column_view& object_id,
        const column_view& x,
        const column_view& y,
        const column_view& timestamp
    ) except +

    cdef unique_ptr[table] trajectory_bounding_boxes(
        size_type num_trajectories,
        const column_view& object_id,
        const column_view& x,
        const column_view& y
    ) except +

    cdef unique_ptr[table] trajectory_distances_and_speeds(
        size_type num_trajectories,
        const column_view& object_id,
        const column_view& x,
        const column_view& y,
        const column_view& timestamp
    ) except +
