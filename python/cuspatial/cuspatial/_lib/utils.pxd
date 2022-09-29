# Copyright (c) 2022, NVIDIA CORPORATION.
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.optional cimport nullopt, optional


cdef optional[column_view] unwrap_pyoptcol(object pyoptcol) except*
