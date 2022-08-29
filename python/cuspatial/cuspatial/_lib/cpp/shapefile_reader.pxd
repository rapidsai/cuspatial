# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column

ctypedef bool winding_order_type_t

cdef extern from "cuspatial/shapefile_reader.hpp" namespace "cuspatial" nogil:
    cdef vector[unique_ptr[column]] \
        read_polygon_shapefile(
            const string filename, winding_order outer_ring_winding
    ) except +
    ctypedef enum winding_order "cuspatial::winding_order":
        COUNTER_CLOCKWISE "cuspatial::winding_order::COUNTER_CLOCKWISE",
        CLOCKWISE "cuspatial::winding_order::CLOCKWISE"
