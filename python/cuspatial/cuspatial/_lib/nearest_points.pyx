from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.optional cimport optional
from cuspatial._lib.cpp.point_linestring_nearest_points cimport (
    pairwise_point_linestring_nearest_points as c_func,
    point_linestring_nearest_points_result,
)
from cuspatial._lib.utils cimport unwrap_pyoptcol


def pairwise_point_linestring_nearest_points(
    Column points_xy,
    Column linestring_part_offsets,
    Column linestring_points_xy,
    multipoint_geometry_offset=None,
    multilinestring_geometry_offset=None,
):
    cdef optional[column_view] c_multipoint_geometry_offset = unwrap_pyoptcol(
        multipoint_geometry_offset)
    cdef optional[column_view] c_multilinestring_geometry_offset = (
        unwrap_pyoptcol(multilinestring_geometry_offset))

    cdef column_view c_points_xy = points_xy.view()
    cdef column_view c_linestring_offsets = linestring_part_offsets.view()
    cdef column_view c_linestring_points_xy = linestring_points_xy.view()
    cdef point_linestring_nearest_points_result c_result

    with nogil:
        c_result = move(c_func(
            c_multipoint_geometry_offset,
            c_points_xy,
            c_multilinestring_geometry_offset,
            c_linestring_offsets,
            c_linestring_points_xy,
        ))

    multipoint_geometry_id = None
    if multipoint_geometry_offset is not None:
        multipoint_geometry_id = Column.from_unique_ptr(
            move(c_result.nearest_point_geometry_id.value()))

    multilinestring_geometry_id = None
    if multilinestring_geometry_offset is not None:
        multilinestring_geometry_id = Column.from_unique_ptr(
            move(c_result.nearest_linestring_geometry_id.value()))

    segment_id = Column.from_unique_ptr(move(c_result.nearest_segment_id))
    point_on_linestring_xy = Column.from_unique_ptr(
        move(c_result.nearest_point_on_linestring_xy))

    return (
        multipoint_geometry_id,
        multilinestring_geometry_id,
        segment_id,
        point_on_linestring_xy
    )
