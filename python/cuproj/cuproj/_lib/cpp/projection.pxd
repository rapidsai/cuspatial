from cuproj._lib.cpp.operation cimport operation_type, direction
from cuproj._lib.cpp.projection_parameters cimport projection_parameters

from libcpp.vector cimport vector

cdef extern from "cuproj/projection.cuh" namespace "cuproj" nogil:
  cdef cppclass projection[Coordinate, T=*]:
    projection()
    projection(vector[operation_type],
               projection_parameters[T],
               direction=direction.FORWARD)

    void transform[CoordIter](CoordIter, CoordIter, CoordIter, direction)
