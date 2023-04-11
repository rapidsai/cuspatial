# Copyright (c) 2023, NVIDIA CORPORATION.

cdef extern from "cuspatial/types.hpp" namespace "cuspatial" nogil:
    cdef enum geometry_type_id:
        POINT "cuspatial::geometry_type_id::POINT"
        LINESTRING "cuspatial::geometry_type_id::LINESTRING"
        POLYGON "cuspatial::geometry_type_id::POLYGON"

    ctypedef enum collection_type_id:
        SINGLE "cuspatial::collection_type_id::SINGLE"
        MULTI "cuspatial::collection_type_id::MULTI"
