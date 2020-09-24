# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cuspatial/shapefile_reader.hpp" namespace "cuspatial" nogil:
    cdef vector[unique_ptr[column]] \
        read_polygon_shapefile(const string filename) except +
