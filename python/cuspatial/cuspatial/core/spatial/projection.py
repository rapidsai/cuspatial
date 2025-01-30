# Copyright (c) 2022-2025, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import ColumnBase

from cuspatial._lib.spatial import (
    sinusoidal_projection as cpp_sinusoidal_projection,
)
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import (
    contain_single_type_geometry,
    contains_only_multipoints,
    contains_only_points,
)


def sinusoidal_projection(origin_lon, origin_lat, lonlat: GeoSeries):
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
    lonlat: GeoSeries
        A GeoSeries of Points that contains the longitude and latitude
        to transform

    Returns
    -------
    result : GeoSeries
        A GeoSeries that contains the transformed coordinates.
    """

    if contain_single_type_geometry(lonlat):
        if not contains_only_points(lonlat) or contains_only_multipoints(
            lonlat
        ):
            raise ValueError("`lonlat` must contain only POINTS geometry.")

    result = cpp_sinusoidal_projection(
        origin_lon,
        origin_lat,
        lonlat.points.x._column.to_pylibcudf(mode="read"),
        lonlat.points.y._column.to_pylibcudf(mode="read"),
    )
    lonlat_transformed = DataFrame(
        {
            "x": ColumnBase.from_pylibcudf(result[0]),
            "y": ColumnBase.from_pylibcudf(result[1]),
        }
    ).interleave_columns()
    return GeoSeries.from_points_xy(lonlat_transformed)
