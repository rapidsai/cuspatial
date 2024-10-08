# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type


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
