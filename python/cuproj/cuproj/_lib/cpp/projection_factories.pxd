from cuproj._lib.cpp.projection cimport projection

from libcpp.string cimport string

cdef extern from "cuproj/projection_factories.hpp" namespace "cuproj" nogil:
    projection[Coordinate] make_projection[Coordinate](
        string, string
    )
