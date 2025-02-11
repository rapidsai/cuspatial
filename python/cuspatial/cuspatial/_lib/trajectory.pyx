# Copyright (c) 2019-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from pylibcudf cimport Column as plc_Column, Table as plc_Table
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type

from cuspatial._lib.cpp.trajectory cimport (
    derive_trajectories as cpp_derive_trajectories,
    trajectory_bounding_boxes as cpp_trajectory_bounding_boxes,
    trajectory_distances_and_speeds as cpp_trajectory_distances_and_speeds,
)


cpdef tuple derive_trajectories(
    plc_Column object_id,
    plc_Column x,
    plc_Column y,
    plc_Column timestamp,
):
    cdef column_view c_id = object_id.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef column_view c_ts = timestamp.view()
    cdef pair[unique_ptr[table], unique_ptr[column]] result
    with nogil:
        result = move(cpp_derive_trajectories(c_id, c_x, c_y, c_ts))
    return (
        plc_Table.from_libcudf(move(result.first)),
        plc_Column.from_libcudf(move(result.second))
    )


cpdef plc_Table trajectory_bounding_boxes(
    size_type num_trajectories,
    plc_Column object_id,
    plc_Column x,
    plc_Column y,
):
    cdef column_view c_id = object_id.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_trajectory_bounding_boxes(
            num_trajectories, c_id, c_x, c_y
        ))
    return plc_Table.from_libcudf(move(result))


cpdef plc_Table trajectory_distances_and_speeds(
    size_type num_trajectories,
    plc_Column object_id, plc_Column x,
    plc_Column y, plc_Column timestamp
):
    cdef column_view c_id = object_id.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef column_view c_ts = timestamp.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_trajectory_distances_and_speeds(
            num_trajectories, c_id, c_x, c_y, c_ts
        ))
    return plc_Table.from_libcudf(move(result))
