# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.column.geometry_column_view cimport (
    geometry_column_view,
)
from cuspatial._lib.cpp.distance.linestring_distance cimport (
    pairwise_linestring_distance as c_pairwise_linestring_distance,
)
from cuspatial._lib.cpp.distance.linestring_polygon_distance cimport (
    pairwise_linestring_polygon_distance as c_pairwise_line_poly_dist,
)
from cuspatial._lib.cpp.distance.point_distance cimport (
    pairwise_point_distance as c_pairwise_point_distance,
)
from cuspatial._lib.cpp.distance.point_linestring_distance cimport (
    pairwise_point_linestring_distance as c_pairwise_point_linestring_distance,
)
from cuspatial._lib.cpp.distance.point_polygon_distance cimport (
    pairwise_point_polygon_distance as c_pairwise_point_polygon_distance,
)
from cuspatial._lib.cpp.optional cimport optional
from cuspatial._lib.cpp.types cimport collection_type_id, geometry_type_id
from cuspatial._lib.types cimport collection_type_py_to_c
from cuspatial._lib.utils cimport unwrap_pyoptcol


def pairwise_point_distance(
    Column points1_xy,
    Column points2_xy,
    multipoint1_offsets=None,
    multipoint2_offsets=None,
):
    cdef optional[column_view] c_multipoints1_offset = unwrap_pyoptcol(
        multipoint1_offsets)
    cdef optional[column_view] c_multipoints2_offset = unwrap_pyoptcol(
        multipoint2_offsets)

    cdef column_view c_points1_xy = points1_xy.view()
    cdef column_view c_points2_xy = points2_xy.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_point_distance(
            c_multipoints1_offset,
            c_points1_xy,
            c_multipoints2_offset,
            c_points2_xy,
        ))
    return Column.from_unique_ptr(move(c_result))


def pairwise_linestring_distance(
    Column linestring1_part_offsets,
    Column linestring1_points_xy,
    Column linestring2_part_offsets,
    Column linestring2_points_xy,
    multilinestring1_geometry_offsets=None,
    multilinestring2_geometry_offsets=None
):
    cdef optional[column_view] c_mls1_geometry_offsets = unwrap_pyoptcol(
        multilinestring1_geometry_offsets)
    cdef optional[column_view] c_mls2_geometry_offsets = unwrap_pyoptcol(
        multilinestring2_geometry_offsets)
    cdef column_view linestring1_offsets_view = linestring1_part_offsets.view()
    cdef column_view linestring1_points_xy_view = linestring1_points_xy.view()
    cdef column_view linestring2_offsets_view = linestring2_part_offsets.view()
    cdef column_view linestring2_points_xy_view = linestring2_points_xy.view()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(c_pairwise_linestring_distance(
            c_mls1_geometry_offsets,
            linestring1_offsets_view,
            linestring1_points_xy_view,
            c_mls2_geometry_offsets,
            linestring2_offsets_view,
            linestring2_points_xy_view,
        ))

    return Column.from_unique_ptr(move(c_result))


def pairwise_point_linestring_distance(
    Column points_xy,
    Column linestring_part_offsets,
    Column linestring_points_xy,
    multipoint_geometry_offset=None,
    multilinestring_geometry_offset=None,
):
    cdef optional[column_view] c_multipoint_parts_offset = unwrap_pyoptcol(
        multipoint_geometry_offset)
    cdef optional[column_view] c_multilinestring_parts_offset = (
        unwrap_pyoptcol(multilinestring_geometry_offset))

    cdef column_view c_points_xy = points_xy.view()
    cdef column_view c_linestring_offsets = linestring_part_offsets.view()
    cdef column_view c_linestring_points_xy = linestring_points_xy.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_point_linestring_distance(
            c_multipoint_parts_offset,
            c_points_xy,
            c_multilinestring_parts_offset,
            c_linestring_offsets,
            c_linestring_points_xy,
        ))
    return Column.from_unique_ptr(move(c_result))


def pairwise_point_polygon_distance(
    point_collection_type,
    Column multipoints,
    Column multipolygons
):

    cdef collection_type_id point_multi_type = collection_type_py_to_c(
        point_collection_type
    )

    cdef shared_ptr[geometry_column_view] c_multipoints = \
        make_shared[geometry_column_view](
            multipoints.view(),
            point_multi_type,
            geometry_type_id.POINT)

    cdef shared_ptr[geometry_column_view] c_multipolygons = \
        make_shared[geometry_column_view](
            multipolygons.view(),
            collection_type_id.MULTI,
            geometry_type_id.POLYGON)

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_point_polygon_distance(
            c_multipoints.get()[0], c_multipolygons.get()[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def pairwise_linestring_polygon_distance(
    Column multilinestrings,
    Column multipolygons
):

    cdef shared_ptr[geometry_column_view] c_multilinestrings = \
        make_shared[geometry_column_view](
            multilinestrings.view(),
            collection_type_id.MULTI,
            geometry_type_id.LINESTRING)

    cdef shared_ptr[geometry_column_view] c_multipolygons = \
        make_shared[geometry_column_view](
            multipolygons.view(),
            collection_type_id.MULTI,
            geometry_type_id.POLYGON)

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_line_poly_dist(
            c_multilinestrings.get()[0], c_multipolygons.get()[0]
        ))

    return Column.from_unique_ptr(move(c_result))
