from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.distance.point_linestring_nearest_point cimport (
    pairwise_point_linestring_nearest_point as c_pairwise_point_linestring_nearest_point,
    point_linestring_nearest_points_result
)
from cuspatial._lib.cpp.optional cimport nullopt, optional


def pairwise_point_linestring_nearest_points(
    Column points_xy,
    Column linestring_part_offsets,
    Column linestring_points_xy,
    multipoint_geometry_offset=None,
    multilinestring_geometry_offset=None,
):
    cdef Column multipoint_geometry_offset
    cdef Column multilinestring_geometry_offset
    cdef optional[column_view] c_multipoint_geometry_offset
    cdef optional[column_view] c_multilinestring_geometry_offset

    if multipoint_geometry_offset is not None:
        c_multipoint_geometry_offset = multipoint_geometry_offset.view()
    else:
        c_multipoint_geometry_offset = nullopt

    if multilinestring_geometry_offset is not None:
        c_multilinestring_geometry_offset = multilinestring_geometry_offset.view()
    else:
        c_multilinestring_geometry_offset = nullopt

    cdef column_view c_points_xy = points_xy.view()
    cdef column_view c_linestring_offsets = linestring_part_offsets.view()
    cdef column_view c_linestring_points_xy = linestring_points_xy.view()
    cdef point_linestring_nearest_points_result c_result

    with nogil:
        c_result = move(c_pairwise_point_linestring_nearest_point(
            c_multipoint_geometry_offset,
            c_points_xy,
            c_multilinestring_geometry_offset,
            c_linestring_offsets,
            c_linestring_points_xy,
        ))

    
    if multipoint_geometry_offset is not None:
        point_geometry_id = Column.from_unique_ptr(move(c_result.nearest_point_geometry_id))
    else:
        point_geometry_id = None
    
    if multilinestring_geometry_offset is not None:
        linestring_geometry_id = Column.from_unique_ptr(move(c_result.nearest_linestring_geometry_id))
    else:
        linestring_geometry_id = None
    
    segment_id = Column.from_unique_ptr(move(c_result.nearest_segment_id))
    point_on_linestring_xy = Column.from_unique_ptr(move(c_result.nearest_point_on_linestring_xy))

    return point_geometry_id, linestring_geometry_id, segment_id, point_on_linestring_xy
