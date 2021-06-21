# Copyright (c) 2021, NVIDIA CORPORATION

from geopandas.geoseries import is_geometry_type as gp_is_geometry_type

from cuspatial.geometry.geoseries import GeoColumn, GeoSeries


def is_geometry_type(obj):
    """
    Returns `True` if the column is a `GeoPandas` or `cuspatial.GeoSeries`
    """
    if isinstance(obj, (GeoSeries, GeoColumn)):
        return True
    if gp_is_geometry_type(obj):
        return True
    return False
