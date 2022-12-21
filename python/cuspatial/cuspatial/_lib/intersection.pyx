# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.utility cimport move

from cudf._lib.column cimport Column

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


def pairwise_linestring_intersection(Column lhs, Column rhs):
    """
    Compute the intersection of two (multi)linestrings.
    """
    from cuspatial.core._column.geometa import FeatureEnum

    cdef linestring_intersection_column_result c_result
    cdef collection_type_id multi_type = <collection_type_id>(
        <underlying_collection_type_id_t>(CollectionType.MULTI.value)
    )
    cdef geometry_type_id linestring_type = <geometry_type_id>(
        <underlying_geometry_type_id_t>(GeometryType.LINESTRING.value)
    )

    cdef geometry_column_view c_lhs = geometry_column_view(
        lhs.view(),
        multi_type,
        linestring_type
    )
    cdef geometry_column_view c_rhs = geometry_column_view(
        rhs.view(),
        multi_type,
        linestring_type
    )

    with nogil:
        c_result = move(cpp_pairwise_linestring_intersection(c_lhs, c_rhs))

    geometry_collection_offset = Column.from_unique_ptr(
        move(c_result.geometry_collection_offset)
    )

    types_buffer = Column.from_unique_ptr(move(c_result.types_buffer))
    offset_buffer = Column.from_unique_ptr(move(c_result.offset_buffer))
    points_xy = Column.from_unique_ptr(move(c_result.points_xy))
    segments_offsets = Column.from_unique_ptr(move(c_result.segments_offsets))
    segments_xy = Column.from_unique_ptr(move(c_result.segments_xy))
    lhs_linestring_id = Column.from_unique_ptr(
        move(c_result.lhs_linestring_id)
    )
    lhs_segment_id = Column.from_unique_ptr(move(c_result.lhs_segment_id))
    rhs_linestring_id = Column.from_unique_ptr(
        move(c_result.rhs_linestring_id)
    )
    rhs_segment_id = Column.from_unique_ptr(move(c_result.rhs_segment_id))

    # Map linestring type codes from libcuspatial to cuspatial
    types_buffer[types_buffer == GeometryType.LINESTRING.value] = (
        FeatureEnum.LINESTRING.value
    )

    return ((geometry_collection_offset,
            types_buffer,
            offset_buffer,
            points_xy,
            segments_offsets,
            segments_xy),
            (lhs_linestring_id,
             lhs_segment_id,
             rhs_linestring_id,
             rhs_segment_id))
