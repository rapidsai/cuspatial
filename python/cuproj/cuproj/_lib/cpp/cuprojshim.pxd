from libcpp.string cimport string

from cuproj._lib.cpp.operation cimport direction
from cuproj._lib.cpp.projection cimport projection


cdef extern from "cuprojshim.hpp" namespace "cuproj" nogil:
    cdef cppclass vec_2d[T]:
        T x
        T y

cdef extern from "cuprojshim.hpp" namespace "cuprojshim" nogil:
    projection[vec_2d[T]]* make_projection[T](string, string) except +

    void transform[T](
        projection[vec_2d[T]],
        vec_2d[T]*,
        vec_2d[T]*,
        size_t,
        direction
    ) except+

    void transform[T](
        projection[vec_2d[T]],
        T*,
        T*,
        T*,
        T*,
        size_t,
        direction
    ) except +
