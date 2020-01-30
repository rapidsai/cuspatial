import numpy as np
from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from cudf.core.buffer import Buffer

cudf_to_np_types = {
    INT8: np.dtype('int8'),
    INT16: np.dtype('int16'),
    INT32: np.dtype('int32'),
    INT64: np.dtype('int64'),
    FLOAT32: np.dtype('float32'),
    FLOAT64: np.dtype('float64'),
    TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    STRING: np.dtype("object"),
    BOOL8: np.dtype("bool")
}

def cpp_quadtree_on_points(Column x, Column y,
	double x1,double y1,double x2,double y2,double scale, 
	int num_levels, int min_size):
    cdef unique_ptr[table] c_result = move(
        quadtree_on_points(x.mutable_view(), y.mutable_view(),x1,y1,x2,y2,scale,num_levels,min_size)            
    )
    return Table.from_unique_ptr(move(c_result), ["x", "y"])

def cpp_nested_column_test(Column x, Column y):
    print('in cpp_nested_column_test..................')
    cdef unique_ptr[column] c_result = move(
        nested_column_test(x.view(),y.view())
    )
    #return  Column from_ptr(move(c_result))
    
    #copied from cudf._libxx.column.pyx for debugging purpose
    from cudf.core.column import build_column
    c_col=move(c_result)
    size = c_col.get()[0].size()
    dtype = cudf_to_np_types[c_col.get()[0].type().id()]
    has_nulls = c_col.get()[0].has_nulls()

    #After call to release(), c_col is unusable
    cdef column_contents contents = c_col.get()[0].release()

    data = DeviceBuffer.c_from_unique_ptr(move(contents.data))
    data = Buffer(data)

    if has_nulls:
        mask = DeviceBuffer.c_from_unique_ptr(move(contents.null_mask))
        mask = Buffer(mask)
    else:
        mask = None

    cdef vector[unique_ptr[column]] c_children = move(contents.children)
    
    print('c_children.size=',c_children.size())
    #not sure about the logic; seems no child column was appended
    children = None
    if c_children.size() != 0:
        children = tuple(Column.from_unique_ptr(move(c_children[i]))
                             for i in range(c_children.size()))
    print('children.size=',children.size())
    return build_column(data, dtype=dtype, mask=mask, children=children)
