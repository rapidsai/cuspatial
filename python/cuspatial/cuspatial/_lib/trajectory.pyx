# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from pylibcudf cimport Table as plc_Table
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type

from cuspatial._lib.cpp.trajectory cimport (
    derive_trajectories as cpp_derive_trajectories,
    trajectory_bounding_boxes as cpp_trajectory_bounding_boxes,
    trajectory_distances_and_speeds as cpp_trajectory_distances_and_speeds,
)


cpdef derive_trajectories(Column object_id, Column x,
                          Column y, Column timestamp):
    cdef column_view c_id = object_id.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef column_view c_ts = timestamp.view()
    cdef pair[unique_ptr[table], unique_ptr[column]] result
    with nogil:
        result = move(cpp_derive_trajectories(c_id, c_x, c_y, c_ts))
    cdef plc_Table plc_table = plc_Table.from_libcudf(move(result.first))
    first_result = (
        {
            name: Column.from_pylibcudf(col)
            for name, col in zip(
                ["object_id", "x", "y", "timestamp"], plc_table.columns()
            )
        },
        None
    )
    return first_result, Column.from_unique_ptr(move(result.second))


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
    cdef plc_Table plc_table = plc_Table.from_libcudf(move(result))
    return (
        {
            name: Column.from_pylibcudf(col)
            for name, col in zip(
                ["x_min", "y_min", "x_max", "y_max"], plc_table.columns()
            )
        },
        None
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
    cdef plc_Table plc_table = plc_Table.from_libcudf(move(result))
    return (
        {
            name: Column.from_pylibcudf(col)
            for name, col in zip(["distance", "speed"], plc_table.columns())
        },
        None
    )
