# Copyright (c) 2020-2021, NVIDIA CORPORATION

from typing import TypeVar, Union

import geopandas as gpd
import pandas as pd
from geopandas.geoseries import GeoSeries as gpGeoSeries

import cudf

from cuspatial.geometry.geoarrowbuffers import GeoArrowBuffers
from cuspatial.geometry.geocolumn import GeoColumn, GeoMeta
from cuspatial.io.geopandas_adapter import GeoPandasAdapter

T = TypeVar("T", bound="GeoSeries")


class GeoSeries(cudf.Series):
    """
    cuspatial.GeoSeries enables GPU-backed storage and computation of
    shapely-like objects. Our goal is to give feature parity with GeoPandas.
    At this time, only from_geopandas and to_geopandas are directly supported.
    cuspatial GIS, indexing, and trajectory functions depend on the arrays
    stored in the `GeoArrowBuffers` object, accessible with the `points`,
    `multipoints`, `lines`, and `polygons` accessors.

    >>> cuseries.points
        xy:
        0   -1.0
        1    0.0
        dtype: float64
    """

    def __init__(
        self,
        data: Union[gpd.GeoSeries],
        index: Union[cudf.Index, pd.Index] = None,
        dtype=None,
        name=None,
        nan_as_null=True,
    ):
        # Condition index
        if isinstance(data, (gpGeoSeries, GeoSeries)):
            if index is None:
                index = data.index
        if index is None:
            index = cudf.RangeIndex(0, len(data))
        # Condition data
        if isinstance(data, pd.Series):
            data = gpGeoSeries(data)
        # Create column
        if isinstance(data, GeoColumn):
            column = data
        elif isinstance(data, GeoSeries):
            column = data._column
        elif isinstance(data, gpGeoSeries):
            adapter = GeoPandasAdapter(data)
            buffers = GeoArrowBuffers(adapter.get_geoarrow_host_buffers())
            pandas_meta = GeoMeta(adapter.get_geopandas_meta())
            column = GeoColumn(buffers, pandas_meta)
        else:
            raise TypeError(
                f"Incompatible object passed to GeoSeries ctor {type(data)}"
            )
        super().__init__(column, index, dtype, name, nan_as_null)

    @property
    def _geocolumn(self):
        """
        The GeoColumn object keeps a reference to a `GeoArrowBuffers` object,
        which contains all of the geometry coordinates and offsets for thie
        `GeoSeries`.
        """
        return self._column

    @_geocolumn.setter
    def _geocolumn(self, value):
        if not isinstance(value, GeoColumn):
            raise TypeError
        self._column = value

    @property
    def points(self):
        """
        Access the `PointsArray` of the underlying `GeoArrowBuffers`.
        """
        return self._geocolumn.points

    @property
    def multipoints(self):
        """
        Access the `MultiPointArray` of the underlying `GeoArrowBuffers`.
        """
        return self._geocolumn.multipoints

    @property
    def lines(self):
        """
        Access the `LineArray` of the underlying `GeoArrowBuffers`.
        """
        return self._geocolumn.lines

    @property
    def polygons(self):
        """
        Access the `PolygonArray` of the underlying `GeoArrowBuffers`.
        """
        return self._geocolumn.polygons

    def __repr__(self):
        # TODO: Implement Iloc with slices so that we can use `Series.__repr__`
        return self.to_pandas().__repr__()

    def to_geopandas(self, nullable=False):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        if nullable is True:
            raise ValueError("GeoSeries doesn't support <NA> yet")
        host_column = self._geocolumn.to_host()
        output = [host_column[i].to_shapely() for i in range(len(host_column))]
        return gpGeoSeries(output, index=self.index.to_pandas())

    def to_pandas(self):
        """
        Treats to_pandas and to_geopandas as the same call, which improves
        compatibility with pandas.
        """
        return self.to_geopandas()
