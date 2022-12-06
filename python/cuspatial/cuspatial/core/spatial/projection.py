# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import DataFrame

from cuspatial._lib.spatial import (
    sinusoidal_projection as cpp_sinusoidal_projection,
)


def sinusoidal_projection(origin_lon, origin_lat, input_lon, input_lat):
    """
    Sinusoidal projection of longitude/latitude relative to origin to
    Cartesian (x/y) coordinates in km.

    Can be used to approximately convert longitude/latitude coordinates
    to Cartesian coordinates given that all points are near the origin.
    Error increases with distance from the origin. Results are scaled
    relative to the size of the Earth in kilometers. See
    https://en.wikipedia.org/wiki/Sinusoidal_projection for more detail.

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

    result = cpp_sinusoidal_projection(
        origin_lon, origin_lat, input_lon._column, input_lat._column
    )
    return DataFrame({"x": result[0], "y": result[1]})
