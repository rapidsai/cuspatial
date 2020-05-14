# Copyright (c) 2019, NVIDIA CORPORATION.


from libc.stdlib cimport malloc, free
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from cudf import Series
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.legacy.cudf cimport *
from cudf._lib.legacy.cudf import *
from cuspatial._lib.cpp.coordinate_transform cimport (
    lonlat_to_cartesian as cpp_lonlat_to_cartesian
)

from cuspatial._lib.cpp.spatial cimport (
    haversine_distance as cpp_haversine_distance
)

from cuspatial._lib.move cimport move

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


cpdef cpp_spatial_window_points(left, bottom, right, top, x, y):
    left = np.float64(left)
    bottom = np.float64(bottom)
    right = np.float64(right)
    top = np.float64(top)
    x = x.astype('float64')._column
    y = y.astype('float64')._column
    cdef gdf_scalar* c_left = gdf_scalar_from_scalar(left)
    cdef gdf_scalar* c_bottom = gdf_scalar_from_scalar(bottom)
    cdef gdf_scalar* c_right = gdf_scalar_from_scalar(right)
    cdef gdf_scalar* c_top = gdf_scalar_from_scalar(top)

    cdef gdf_column* c_x = column_view_from_column(x)
    cdef gdf_column* c_y = column_view_from_column(y)

    cdef gdf_column* c_out_x = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_out_y = <gdf_column*>malloc(sizeof(gdf_column))

    cdef pair[gdf_column, gdf_column] xy

    with nogil:
        xy = spatial_window_points(
            c_left[0],
            c_bottom[0],
            c_right[0],
            c_top[0],
            c_x[0],
            c_y[0]
        )

    return (Series(gdf_column_to_column(&xy.first)),
            Series(gdf_column_to_column(&xy.second)))
