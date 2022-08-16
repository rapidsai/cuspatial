# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import DataFrame
from cuspatial._lib.spatial import (
    lonlat_to_cartesian as cpp_lonlat_to_cartesian,
)

def lonlat_to_cartesian(origin_lon, origin_lat, input_lon, input_lat):
    """
    Convert lon/lat to ``x,y`` coordinates with respect to an origin lon/lat
    point. Results are scaled relative to the size of the Earth in kilometers.

    Parameters
    ----------
    origin_lon : ``number``
        longitude offset  (this is subtracted from each input before
        converting to x,y)
    origin_lat : ``number``
        latitude offset (this is subtracted from each input before
        converting to x,y)
    input_lon : ``Series`` or ``list``
        longitude coordinates to convert to x
    input_lat : ``Series`` or ``list``
        latitude coordinates to convert to y

    Returns
    -------
    result : cudf.DataFrame
        x : cudf.Series
            x-coordinate of the input relative to the size of the Earth in
            kilometers.
        y : cudf.Series
            y-coordinate of the input relative to the size of the Earth in
            kilometers.
    """
    result = cpp_lonlat_to_cartesian(
        origin_lon, origin_lat, input_lon._column, input_lat._column
    )
    return DataFrame({"x": result[0], "y": result[1]})
