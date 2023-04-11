# Copyright (c) 2022, NVIDIA CORPORATION.

from enum import IntEnum

from cuspatial._lib.cpp.types cimport collection_type_id, geometry_type_id
from cuspatial._lib.types cimport (
    underlying_collection_type_id_t,
    underlying_geometry_type_id_t,
)


class GeometryType(IntEnum):
    POINT = (
        <underlying_geometry_type_id_t> geometry_type_id.POINT
    )
    LINESTRING = (
        <underlying_geometry_type_id_t> geometry_type_id.LINESTRING
    )
    POLYGON = (
        <underlying_geometry_type_id_t> geometry_type_id.POLYGON
    )


class CollectionType(IntEnum):
    SINGLE = (
        <underlying_collection_type_id_t> collection_type_id.SINGLE
    )
    MULTI = (
        <underlying_collection_type_id_t> collection_type_id.MULTI
    )


cdef geometry_type_id geometry_type_py_to_c(typ : GeometryType):
    return <geometry_type_id>(<underlying_collection_type_id_t> (typ.value))

cdef collection_type_id collection_type_py_to_c(typ : CollectionType):
    return <collection_type_id>(<underlying_collection_type_id_t> (typ.value))
