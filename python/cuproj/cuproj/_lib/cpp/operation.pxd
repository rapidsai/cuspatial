
cdef extern from "cuproj/operation/operation.cuh" namespace "cuproj" nogil:
    cdef enum operation_type:
        AXIS_SWAP = 0
        DEGREES_TO_RADIANS = 1
        CLAMP_ANGULAR_COORDINATES = 2
        OFFSET_SCALE_CARTESIAN_COORDINATES = 3
        TRANSVERSE_MERCATOR = 4

    cdef enum direction:
        FORWARD = 0
        INVERSE = 1
