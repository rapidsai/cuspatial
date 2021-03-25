# Copyright (c) 2020-2021, NVIDIA CORPORATION

import pandas as pd

from geopandas.geoseries import GeoSeries as gpGeoSeries

from typing import (
    TypeVar,
    Union,
)

import cudf
import geopandas as gpd

from cuspatial.io.geopandas_adapter import GeoPandasAdapter
from cuspatial.geometry.geocolumn import GeoColumn, GeoPandasMeta
from cuspatial.geometry.geoarrowbuffers import GeoArrowBuffers


T = TypeVar("T", bound="GeoSeries")


class GeoSeries(cudf.Series):
    """
    A wrapper for Series functionality that needs to be subclassed in order
    to support our implementation of GeoColumn.
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
            pandas_meta = GeoPandasMeta(adapter.get_geopandas_meta())
            column = GeoColumn(buffers, pandas_meta)
        else:
            raise TypeError(
                f"Incompatible object passed to GeoSeries ctor {type(data)}"
            )
        super().__init__(column, index, dtype, name, nan_as_null)

    @property
    def geocolumn(self):
        return self._column

    @geocolumn.setter
    def geocolumn(self, value):
        if not isinstance(value, GeoColumn):
            raise TypeError
        self._column = value

    @property
    def points(self):
        """
        The Points column is a simple numeric column. x and y coordinates
        can be stored either interleaved or in separate columns. If a z
        coordinate is present, it will be stored in a separate column.
        """
        return self.geocolumn.points

    @property
    def multipoints(self):
        """
        The MultiPoints column is similar to the Points column with the
        addition of an offsets column. The offsets column stores the comparable
        sizes and coordinates of each MultiPoint in the cuGeoSeries.
        """
        return self.geocolumn.multipoints

    @property
    def lines(self):
        """
        LineStrings contain the coordinates column, an offsets column, and a
        multioffsets column. The multioffsets column stores the indices of the
        offsets that indicate the beginning and end of each MultiLineString
        segment.
        """
        return self.geocolumn.lines

    @property
    def polygons(self):
        """
        Polygons contain the coordinates column, a rings column specifying
        the beginning and end of every polygon, a polygons column specifying
        the beginning, or exterior, ring of each polygon and the end ring.
        All rings after the first ring are interior rings.  Finally a
        multipolys column stores the offsets of the polygons that should be
        grouped into MultiPolygons.
        """
        return self.geocolumn.polygons

    def __getitem__(self, key):
        result = self._column[key]
        return result

    def __repr__(self):
        # TODO: Limit the the number of rows like cudf does
        return self.to_pandas().__repr__()

    def to_geopandas(self, nullable=False):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        if nullable is True:
            raise ValueError("cuGeoSeries doesn't support <NA> yet")
        output = [geom.to_shapely() for geom in self.geocolumn]
        return gpGeoSeries(output, index=self.index.to_pandas())

    def to_pandas(self):
        """
        Treats to_pandas and to_geopandas as the same call, which improves
        compatibility with pandas.
        """
        return self.to_geopandas()
