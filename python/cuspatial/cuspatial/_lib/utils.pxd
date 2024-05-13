# Copyright (c) 2022-2024, NVIDIA CORPORATION.
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view

from cuspatial._lib.cpp.optional cimport nullopt, optional


cdef optional[column_view] unwrap_pyoptcol(object pyoptcol) except*
