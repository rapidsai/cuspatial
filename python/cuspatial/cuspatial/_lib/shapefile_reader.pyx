# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column, column

from cuspatial._lib.cpp.shapefile_reader cimport (
    read_polygon_shapefile as cpp_read_polygon_shapefile,
    winding_order,
)


cpdef read_polygon_shapefile(object filepath, object winding):
    cdef string c_string = str(filepath).encode()
    cdef vector[unique_ptr[column]] c_result
    winding_map = {
        0: winding_order.COUNTER_CLOCKWISE,
        1: winding_order.CLOCKWISE
    }
    cdef winding_order winding_value = winding_map[winding.value]
    with nogil:
        c_result = move(cpp_read_polygon_shapefile(c_string, winding_value))
    return (
        Column.from_unique_ptr(move(c_result[0])),
        Column.from_unique_ptr(move(c_result[1])),
        Column.from_unique_ptr(move(c_result[2])),
        Column.from_unique_ptr(move(c_result[3])),
    )
