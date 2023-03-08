# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column, column_view

from cuspatial._lib.cpp.types cimport collection_type_id, geometry_type_id


cdef extern from "cuspatial/column/geometry_column_view.hpp" \
        namespace "cuspatial" nogil:

    cdef cppclass geometry_column_view:
        geometry_column_view() except +

        geometry_column_view(
            const column_view& column,
            collection_type_id collection_type,
            geometry_type_id geometry_type
        ) except +
