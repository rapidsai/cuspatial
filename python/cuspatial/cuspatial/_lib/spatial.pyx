# Copyright (c) 2019-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from pylibcudf cimport Column as plc_Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from cuspatial._lib.cpp.projection cimport (
    sinusoidal_projection as cpp_sinusoidal_projection,
)


def sinusoidal_projection(
    double origin_lon,
    double origin_lat,
    plc_Column input_lon,
    plc_Column input_lat
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

    return (
        plc_Column.from_libcudf(move(result.first)),
        plc_Column.from_libcudf(move(result.second))
    )
