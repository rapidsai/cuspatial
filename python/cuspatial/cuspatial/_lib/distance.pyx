# Copyright (c) 2022-2025, NVIDIA CORPORATION.

from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move, pair

from pylibcudf cimport Column as plc_Column, Table as plc_Table
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view

from cuspatial._lib.cpp.column.geometry_column_view cimport (
    geometry_column_view,
)
from cuspatial._lib.cpp.distance cimport (
    directed_hausdorff_distance as directed_cpp_hausdorff_distance,
    haversine_distance as cpp_haversine_distance,
    pairwise_linestring_distance as c_pairwise_linestring_distance,
    pairwise_linestring_polygon_distance as c_pairwise_line_poly_dist,
    pairwise_point_distance as c_pairwise_point_distance,
    pairwise_point_linestring_distance as c_pairwise_point_linestring_distance,
    pairwise_point_polygon_distance as c_pairwise_point_polygon_distance,
    pairwise_polygon_distance as c_pairwise_polygon_distance,
)
from cuspatial._lib.cpp.types cimport collection_type_id, geometry_type_id
from cuspatial._lib.types cimport collection_type_py_to_c


cpdef plc_Column haversine_distance(
    plc_Column x1,
    plc_Column y1,
    plc_Column x2,
    plc_Column y2,
):
    cdef column_view c_x1 = x1.view()
    cdef column_view c_y1 = y1.view()
    cdef column_view c_x2 = x2.view()
    cdef column_view c_y2 = y2.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_haversine_distance(c_x1, c_y1, c_x2, c_y2))

    return plc_Column.from_libcudf(move(c_result))


def directed_hausdorff_distance(
    plc_Column xs,
    plc_Column ys,
    plc_Column space_offsets,
):
    cdef column_view c_xs = xs.view()
    cdef column_view c_ys = ys.view()
    cdef column_view c_shape_offsets = space_offsets.view()

    cdef pair[unique_ptr[column], table_view] result

    with nogil:
        result = move(
            directed_cpp_hausdorff_distance(
                c_xs,
                c_ys,
                c_shape_offsets,
            )
        )

    cdef plc_Column owner_col = plc_Column.from_libcudf(move(result.first))
    cdef plc_Table plc_owner_table = plc_Table(
        [owner_col] * result.second.num_columns()
    )
    cdef plc_Table plc_result_table = plc_Table.from_table_view(
        result.second, plc_owner_table
    )
    return plc_result_table.columns()


def pairwise_point_distance(
    lhs_point_collection_type,
    rhs_point_collection_type,
    plc_Column points1,
    plc_Column points2,
):
    cdef collection_type_id lhs_point_multi_type = collection_type_py_to_c(
        lhs_point_collection_type
    )
    cdef collection_type_id rhs_point_multi_type = collection_type_py_to_c(
        rhs_point_collection_type
    )
    cdef shared_ptr[geometry_column_view] c_multipoints_lhs = \
        make_shared[geometry_column_view](
            points1.view(),
            lhs_point_multi_type,
            geometry_type_id.POINT)
    cdef shared_ptr[geometry_column_view] c_multipoints_rhs = \
        make_shared[geometry_column_view](
            points2.view(),
            rhs_point_multi_type,
            geometry_type_id.POINT)

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_point_distance(
            c_multipoints_lhs.get()[0],
            c_multipoints_rhs.get()[0],
        ))
    return plc_Column.from_libcudf(move(c_result))


def pairwise_linestring_distance(
    plc_Column multilinestrings1,
    plc_Column multilinestrings2
):
    cdef shared_ptr[geometry_column_view] c_multilinestring_lhs = \
        make_shared[geometry_column_view](
            multilinestrings1.view(),
            collection_type_id.MULTI,
            geometry_type_id.LINESTRING)
    cdef shared_ptr[geometry_column_view] c_multilinestring_rhs = \
        make_shared[geometry_column_view](
            multilinestrings2.view(),
            collection_type_id.MULTI,
            geometry_type_id.LINESTRING)

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_linestring_distance(
            c_multilinestring_lhs.get()[0],
            c_multilinestring_rhs.get()[0],
        ))

    return plc_Column.from_libcudf(move(c_result))


def pairwise_point_linestring_distance(
    point_collection_type,
    plc_Column points,
    plc_Column linestrings,
):
    cdef collection_type_id points_multi_type = collection_type_py_to_c(
        point_collection_type
    )
    cdef shared_ptr[geometry_column_view] c_points = \
        make_shared[geometry_column_view](
            points.view(),
            points_multi_type,
            geometry_type_id.POINT)
    cdef shared_ptr[geometry_column_view] c_multilinestrings = \
        make_shared[geometry_column_view](
            linestrings.view(),
            collection_type_id.MULTI,
            geometry_type_id.LINESTRING)

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_point_linestring_distance(
            c_points.get()[0],
            c_multilinestrings.get()[0],
        ))

    return plc_Column.from_libcudf(move(c_result))


def pairwise_point_polygon_distance(
    point_collection_type,
    plc_Column multipoints,
    plc_Column multipolygons
):
    cdef collection_type_id points_multi_type = collection_type_py_to_c(
        point_collection_type
    )

    cdef shared_ptr[geometry_column_view] c_multipoints = \
        make_shared[geometry_column_view](
            multipoints.view(),
            points_multi_type,
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

    return plc_Column.from_libcudf(move(c_result))


def pairwise_linestring_polygon_distance(
    plc_Column multilinestrings,
    plc_Column multipolygons
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

    return plc_Column.from_libcudf(move(c_result))


def pairwise_polygon_distance(plc_Column lhs, plc_Column rhs):
    cdef shared_ptr[geometry_column_view] c_lhs = \
        make_shared[geometry_column_view](
            lhs.view(),
            collection_type_id.MULTI,
            geometry_type_id.POLYGON)

    cdef shared_ptr[geometry_column_view] c_rhs = \
        make_shared[geometry_column_view](
            rhs.view(),
            collection_type_id.MULTI,
            geometry_type_id.POLYGON)

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_polygon_distance(
            c_lhs.get()[0], c_rhs.get()[0]
        ))

    return plc_Column.from_libcudf(move(c_result))
