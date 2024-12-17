# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move, pair

from cudf._lib.column cimport Column
from pylibcudf cimport Table as plc_Table
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


cpdef haversine_distance(Column x1, Column y1, Column x2, Column y2):
    cdef column_view c_x1 = x1.view()
    cdef column_view c_y1 = y1.view()
    cdef column_view c_x2 = x2.view()
    cdef column_view c_y2 = y2.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_haversine_distance(c_x1, c_y1, c_x2, c_y2))

    return Column.from_unique_ptr(move(c_result))


def directed_hausdorff_distance(
    Column xs,
    Column ys,
    Column space_offsets,
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

    owner_col = Column.from_unique_ptr(
        move(result.first), data_ptr_exposed=True
    )
    cdef plc_Table plc_owner_table = plc_Table(
        [owner_col.to_pylibcudf(mode="read")] * result.second.num_columns()
    )
    cdef plc_Table plc_result_table = plc_Table.from_table_view(
        result.second, plc_owner_table
    )
    return [Column.from_pylibcudf(col) for col in plc_result_table.columns()]


def pairwise_point_distance(
    lhs_point_collection_type,
    rhs_point_collection_type,
    Column points1,
    Column points2,
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
    return Column.from_unique_ptr(move(c_result))


def pairwise_linestring_distance(
    Column multilinestrings1,
    Column multilinestrings2
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

    return Column.from_unique_ptr(move(c_result))


def pairwise_point_linestring_distance(
    point_collection_type,
    Column points,
    Column linestrings,
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

    return Column.from_unique_ptr(move(c_result))


def pairwise_point_polygon_distance(
    point_collection_type,
    Column multipoints,
    Column multipolygons
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


def pairwise_polygon_distance(Column lhs, Column rhs):
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

    return Column.from_unique_ptr(move(c_result))
