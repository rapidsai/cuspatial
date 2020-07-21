from cudf._lib.column cimport column, column_view, Column
from cudf._lib.move cimport move, unique_ptr

from cuspatial._lib.cpp.polygon_separation \
    cimport directed_polygon_separation as cpp_directed_polygon_separation


def directed_polygon_separation(
    Column xs,
    Column ys,
    Column space_offsets
):
    cdef column_view c_xs = xs.view()
    cdef column_view c_ys = ys.view()
    cdef column_view c_space_offsets = space_offsets.view()

    cdef unique_ptr[column] result

    with nogil:
        result = move(
            cpp_directed_polygon_separation(
                c_xs,
                c_ys,
                c_space_offsets
            )
        )

    return Column.from_unique_ptr(move(result))
