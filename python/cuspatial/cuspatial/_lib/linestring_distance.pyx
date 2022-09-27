from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.distance.linestring_distance cimport (
    pairwise_linestring_distance as cpp_pairwise_linestring_distance,
)


def pairwise_linestring_distance(
    Column linestring1_offsets,
    Column linestring1_points_x,
    Column linestring1_points_y,
    Column linestring2_offsets,
    Column linestring2_points_x,
    Column linestring2_points_y
):
    cdef column_view linestring1_offsets_view = linestring1_offsets.view()
    cdef column_view linestring1_points_x_view = linestring1_points_x.view()
    cdef column_view linestring1_points_y_view = linestring1_points_y.view()
    cdef column_view linestring2_offsets_view = linestring2_offsets.view()
    cdef column_view linestring2_points_x_view = linestring2_points_x.view()
    cdef column_view linestring2_points_y_view = linestring2_points_y.view()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_pairwise_linestring_distance(
            linestring1_offsets_view,
            linestring1_points_x_view,
            linestring1_points_y_view,
            linestring2_offsets_view,
            linestring2_points_x_view,
            linestring2_points_y_view
        ))

    return Column.from_unique_ptr(move(c_result))
