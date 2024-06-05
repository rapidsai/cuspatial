# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view

from cuspatial._lib.cpp.projection cimport (
    sinusoidal_projection as cpp_sinusoidal_projection,
)


def sinusoidal_projection(
    double origin_lon,
    double origin_lat,
    Column input_lon,
    Column input_lat
):
    cdef column_view c_input_lon = input_lon.view()
    cdef column_view c_input_lat = input_lat.view()

    cdef pair[unique_ptr[column], unique_ptr[column]] result

    with nogil:
        result = move(
            cpp_sinusoidal_projection(
                origin_lon,
                origin_lat,
                c_input_lon,
                c_input_lat
            )
        )

    return (Column.from_unique_ptr(move(result.first)),
            Column.from_unique_ptr(move(result.second)))
