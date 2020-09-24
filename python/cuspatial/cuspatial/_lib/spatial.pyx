# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import Series
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cuspatial._lib.cpp.coordinate_transform cimport (
    lonlat_to_cartesian as cpp_lonlat_to_cartesian
)

from cuspatial._lib.cpp.spatial cimport (
    haversine_distance as cpp_haversine_distance
)

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

cpdef haversine_distance(Column x1, Column y1, Column x2, Column y2):
    cdef column_view c_x1 = x1.view()
    cdef column_view c_y1 = y1.view()
    cdef column_view c_x2 = x2.view()
    cdef column_view c_y2 = y2.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_haversine_distance(c_x1, c_y1, c_x2, c_y2))

    return Column.from_unique_ptr(move(c_result))


def lonlat_to_cartesian(
    double origin_lon,
    double origin_lat,
    Column input_lon,
    Column input_lat
):
    cdef column_view c_input_lon = input_lon.view()
    cdef column_view c_input_lat = input_lat.view()

    cdef pair[unique_ptr[column], unique_ptr[column]] result

    with nogil:
        result = move(
            cpp_lonlat_to_cartesian(
                origin_lon,
                origin_lat,
                c_input_lon,
                c_input_lat
            )
        )

    return (Column.from_unique_ptr(move(result.first)),
            Column.from_unique_ptr(move(result.second)))
