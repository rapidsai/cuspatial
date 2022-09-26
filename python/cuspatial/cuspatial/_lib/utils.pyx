# Copyright (c) 2022, NVIDIA CORPORATION.
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.optional cimport nullopt, optional


cdef optional[column_view] unwrap_pyoptcol(pyoptcol):
    # Unwrap python optional Column arg to c optional[column_view]
    cdef Column pycol
    cdef optional[column_view] c_opt = nullopt
    if isinstance(pyoptcol, Column):
        pycol = pyoptcol
        c_opt = pycol.view()
    else:
        raise ValueError("pyoptcol must be a Column or None")
    return c_opt
