from cuproj._lib.cpp.ellipsoid cimport ellipsoid


cdef extern from "cuproj/projection_parameters.hpp" namespace "cuproj" nogil:
    cdef enum hemisphere:
        NORTH = 0
        SOUTH = 1

    cdef cppclass projection_parameters[T]:
        projection_parameters(
            ellipsoid[T],
            int,
            hemisphere,
            T,
            T
        ) except+
