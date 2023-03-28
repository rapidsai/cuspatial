# Copyright (c) 2023, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t

from cuspatial._lib.cpp.types cimport collection_type_id, geometry_type_id

ctypedef uint8_t underlying_geometry_type_id_t

ctypedef uint8_t underlying_collection_type_id_t


cdef geometry_type_id geometry_type_py_to_c(typ) except*

cdef collection_type_id collection_type_py_to_c(typ) except*
