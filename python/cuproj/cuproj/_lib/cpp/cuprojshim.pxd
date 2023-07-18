from cuproj._lib.cpp.projection cimport projection
from cuproj._lib.cpp.operation cimport direction

cdef extern from "cuprojshim.hpp" namespace "cuproj" nogil:
    void transform(
        projection[double],
        vec_2d*,
        vec_2d*,
        size_t,
        direction
    )
