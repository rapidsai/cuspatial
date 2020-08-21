# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from cudf._lib.cpp.column.column cimport column

cdef extern from "cuspatial/shapefile_reader.hpp" namespace "cuspatial" nogil:
    cdef vector[unique_ptr[column]] \
        read_polygon_shapefile(const string filename) except +
