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
from cuspatial._lib.move cimport move

cpdef cpp_point_in_polygon_bitmap(
    points_x, points_y, poly_fpos, poly_rpos, poly_x, poly_y
):
    points_x = points_x.astype('float64')._column
    points_y = points_y.astype('float64')._column
    poly_fpos = poly_fpos.astype('int32')._column
    poly_rpos = poly_rpos.astype('int32')._column
    poly_x = poly_x.astype('float64')._column
    poly_y = poly_y.astype('float64')._column
    cdef gdf_column* c_points_x = column_view_from_column(points_x)
    cdef gdf_column* c_points_y = column_view_from_column(points_y)

    cdef gdf_column* c_poly_fpos = column_view_from_column(poly_fpos)
    cdef gdf_column* c_poly_rpos = column_view_from_column(poly_rpos)

    cdef gdf_column* c_poly_x = column_view_from_column(poly_x)
    cdef gdf_column* c_poly_y = column_view_from_column(poly_y)
    cdef gdf_column* result_bitmap = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        result_bitmap[0] = point_in_polygon_bitmap(
            c_points_x[0],
            c_points_y[0],
            c_poly_fpos[0],
            c_poly_rpos[0],
            c_poly_x[0],
            c_poly_y[0]
        )

    free(c_points_x)
    free(c_points_y)
    free(c_poly_fpos)
    free(c_poly_rpos)
    free(c_poly_x)
    free(c_poly_y)
    result = gdf_column_to_column(result_bitmap)
    free(result_bitmap)

    return result

cpdef cpp_haversine_distance(x1, y1, x2, y2):
    x1 = x1.astype('float64')._column
    y1 = y1.astype('float64')._column
    x2 = x2.astype('float64')._column
    y2 = y2.astype('float64')._column

    cdef gdf_column* c_x1 = column_view_from_column(x1)
    cdef gdf_column* c_y1 = column_view_from_column(y1)
    cdef gdf_column* c_x2 = column_view_from_column(x2)
    cdef gdf_column* c_y2 = column_view_from_column(y2)

    cdef gdf_column* c_h_dist = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        c_h_dist[0] = haversine_distance(
            c_x1[0],
            c_y1[0],
            c_x2[0],
            c_y2[0]
        )

    free(c_x1)
    free(c_y1)
    free(c_x2)
    free(c_y2)

    result = Series(gdf_column_to_column(c_h_dist))
    free(c_h_dist)

    return result


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


cpdef cpp_directed_hausdorff_distance(coor_x, coor_y, cnt):
    coor_x = coor_x.astype('float64')._column
    coor_y = coor_y.astype('float64')._column
    cnt = cnt.astype('int32')._column
    cdef gdf_column* c_coor_x = column_view_from_column(coor_x)
    cdef gdf_column* c_coor_y = column_view_from_column(coor_y)
    cdef gdf_column* c_cnt = column_view_from_column(cnt)
    cdef gdf_column* c_dist = <gdf_column*>malloc(sizeof(gdf_column))
    with nogil:
        c_dist[0] = directed_hausdorff_distance(
            c_coor_x[0],
            c_coor_y[0],
            c_cnt[0]
        )

    return Series(gdf_column_to_column(c_dist))

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
