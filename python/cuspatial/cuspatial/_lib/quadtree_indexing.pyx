from cudf._libxx.column cimport *
from cudf._libxx.table cimport *

cimport cudf._libxx.lib as libcudf


def cpp_quadtree_on_points(Column x, Column y):
    cdef unique_ptr[table] c_result = move(
        quadtree_on_points(x.view(), y.view())            
    )
    return _Table.from_ptr(move(c_result))
