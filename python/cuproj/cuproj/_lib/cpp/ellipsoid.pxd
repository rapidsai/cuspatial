cdef extern from "cuproj/ellipsoid.hpp" namespace "cuproj" nogil:
    cdef cppclass ellipsoid[T]:
        ellipsoid() except+
        ellipsoid(T, T) except+
