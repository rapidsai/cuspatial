from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cuproj._lib.cpp.operation cimport direction
from cuproj._lib.cpp.projection cimport projection


cdef extern from "cuprojshim.hpp" namespace "cuproj" nogil:
    cdef cppclass vec_2d[T]:
        T x
        T y

cdef extern from "cuprojshim.hpp" namespace "cuprojshim" nogil:
    projection[vec_2d[double]]* make_projection(string, string) except +

    void transform(
        projection[vec_2d[double]],
        vec_2d[double]*,
        vec_2d[double]*,
        size_t,
        direction
    )

    void transform(
        projection[vec_2d[double]],
        double*,
        double*,
        double*,
        double*,
        size_t,
        direction
    ) except +
