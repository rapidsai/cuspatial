from libcpp.vector cimport vector

from cuproj._lib.cpp.operation cimport direction, operation_type
from cuproj._lib.cpp.projection_parameters cimport projection_parameters


cdef extern from "cuproj/projection.cuh" namespace "cuproj" nogil:
    cdef cppclass projection[Coordinate, T=*]:
        projection()
        projection(vector[operation_type],
                   projection_parameters[T],
                   direction=direction.FORWARD)

        void transform[CoordIter](CoordIter, CoordIter, CoordIter, direction)
