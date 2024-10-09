# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view

from cuspatial._lib.cpp.column.geometry_column_view cimport (
    geometry_column_view,
)


cdef extern from "cuspatial/distance.hpp" \
        namespace "cuspatial" nogil:

    cdef pair[unique_ptr[column], table_view] directed_hausdorff_distance(
        const column_view& xs,
        const column_view& ys,
        const column_view& space_offsets
    ) except +

    cdef unique_ptr[column] haversine_distance(
        const column_view& a_lon,
        const column_view& a_lat,
        const column_view& b_lon,
        const column_view& b_lat
    ) except +

    cdef unique_ptr[column] pairwise_point_distance(
        const geometry_column_view & multipoints1,
        const geometry_column_view & multipoints2
    ) except +

    cdef unique_ptr[column] pairwise_point_linestring_distance(
        const geometry_column_view & multipoints,
        const geometry_column_view & multilinestrings
    ) except +

    cdef unique_ptr[column] pairwise_point_polygon_distance(
        const geometry_column_view & multipoints,
        const geometry_column_view & multipolygons
    ) except +

    cdef unique_ptr[column] pairwise_linestring_distance(
        const geometry_column_view & multilinestrings1,
        const geometry_column_view & multilinestrings2
    ) except +

    cdef unique_ptr[column] pairwise_linestring_polygon_distance(
        const geometry_column_view & multilinestrings,
        const geometry_column_view & multipolygons
    ) except +

    cdef unique_ptr[column] pairwise_polygon_distance(
        const geometry_column_view & multipolygons1,
        const geometry_column_view & multipolygons2
    ) except +
