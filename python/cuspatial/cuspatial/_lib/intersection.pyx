# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from libcpp.memory cimport make_shared, shared_ptr
from libcpp.utility cimport move

from pylibcudf cimport Column as plc_Column

from cuspatial._lib.types import CollectionType, GeometryType

from cuspatial._lib.cpp.column.geometry_column_view cimport (
    geometry_column_view,
)
from cuspatial._lib.cpp.linestring_intersection cimport (
    linestring_intersection_column_result,
    pairwise_linestring_intersection as cpp_pairwise_linestring_intersection,
)
from cuspatial._lib.cpp.types cimport collection_type_id, geometry_type_id
from cuspatial._lib.types cimport (
    underlying_collection_type_id_t,
    underlying_geometry_type_id_t,
)


def pairwise_linestring_intersection(plc_Column lhs, plc_Column rhs):
    """
    Compute the intersection of two (multi)linestrings.
    """
    cdef linestring_intersection_column_result c_result
    cdef collection_type_id multi_type = <collection_type_id>(
        <underlying_collection_type_id_t>(CollectionType.MULTI.value)
    )
    cdef geometry_type_id linestring_type = <geometry_type_id>(
        <underlying_geometry_type_id_t>(GeometryType.LINESTRING.value)
    )

    cdef shared_ptr[geometry_column_view] c_lhs = \
        make_shared[geometry_column_view](
            lhs.view(),
            multi_type,
            linestring_type)
    cdef shared_ptr[geometry_column_view] c_rhs = \
        make_shared[geometry_column_view](
            rhs.view(),
            multi_type,
            linestring_type)

    with nogil:
        c_result = move(cpp_pairwise_linestring_intersection(
            c_lhs.get()[0], c_rhs.get()[0]
        ))

    geometry_collection_offset = plc_Column.from_libcudf(
        move(c_result.geometry_collection_offset)
    )
    types_buffer = plc_Column.from_libcudf(move(c_result.types_buffer))
    offset_buffer = plc_Column.from_libcudf(move(c_result.offset_buffer))
    points = plc_Column.from_libcudf(move(c_result.points))
    segments = plc_Column.from_libcudf(move(c_result.segments))
    lhs_linestring_id = plc_Column.from_libcudf(
        move(c_result.lhs_linestring_id)
    )
    lhs_segment_id = plc_Column.from_libcudf(move(c_result.lhs_segment_id))
    rhs_linestring_id = plc_Column.from_libcudf(
        move(c_result.rhs_linestring_id)
    )
    rhs_segment_id = plc_Column.from_libcudf(move(c_result.rhs_segment_id))

    return ((geometry_collection_offset,
            types_buffer,
            offset_buffer,
            points,
            segments),
            (lhs_linestring_id,
             lhs_segment_id,
             rhs_linestring_id,
             rhs_segment_id))
