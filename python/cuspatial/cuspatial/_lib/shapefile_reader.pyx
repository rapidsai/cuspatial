# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column, column

from cuspatial._lib.cpp.shapefile_reader cimport (
    read_polygon_shapefile as cpp_read_polygon_shapefile,
)


cpdef read_polygon_shapefile(object filepath, bint reversed):
    cdef string c_string = str(filepath).encode()
    cdef vector[unique_ptr[column]] c_result
    with nogil:
        c_result = move(cpp_read_polygon_shapefile(c_string, reversed))
    return (
        Column.from_unique_ptr(move(c_result[0])),
        Column.from_unique_ptr(move(c_result[1])),
        Column.from_unique_ptr(move(c_result[2])),
        Column.from_unique_ptr(move(c_result[3])),
    )
