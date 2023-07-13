
from libcpp.vector cimport vector

cdef extern from "cuproj/projection.cuh" namespace "cuproj" nogil:
  cdef cppclass projection[Coordinate]:
    projection(# vector of operation_type
               # projection parameters
               # constructed direction)
