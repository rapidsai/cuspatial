# Copyright (c) 2020-2021, NVIDIA CORPORATION

import pandas as pd

from geopandas.geoseries import GeoSeries as gpGeoSeries

import cudf

from cuspatial.geometry.geocolumn import GeoColumn


class GeoSeries(cudf.Series):
    """
    A wrapper for Series functionality that needs to be subclassed in order
    to support our implementation of GeoColumn.
    """

    def __init__(
        self, data=None, index=None, dtype=None, name=None, nan_as_null=True
    ):
        if isinstance(data, (gpGeoSeries, GeoSeries)):
            if index is None:
                index = data.index
        if isinstance(data, pd.Series):
            data = gpGeoSeries(data)
        if isinstance(data, (GeoColumn, gpGeoSeries, GeoSeries, dict)):
            column = GeoColumn(data)
            super().__init__(column, index, dtype, name, nan_as_null)
            if isinstance(data, GeoColumn):
                self.geocolumn = data
            else:
                self.geocolumn = GeoColumn(data)
        else:
            raise TypeError(
                f"Incompatible object passed to GeoSeries ctor {type(data)}"
            )

    @property
    def geocolumn(self):
        return self._geocolumn

    @geocolumn.setter
    def geocolumn(self, value):
        if not isinstance(value, GeoColumn):
            raise TypeError
        self._geocolumn = value
        self._column = cudf.RangeIndex(0, len(value))

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

    @property
    def index(self):
        """
        A cudf.Index object. A mapping of row-labels.
        """
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    def __getitem__(self, key):
        result = self.geocolumn[self._column[key]]
        return result

    def __repr__(self):
        # TODO: Limit the the number of rows like cudf does
        return self.to_pandas().__repr__()

    def to_geopandas(self, index=None, nullable=False):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        if nullable is True:
            raise ValueError("cuGeoSeries doesn't support N/A yet")
        if index is None:
            index = self.index
        if isinstance(index, cudf.Index):
            index = index.to_pandas()
        output = [geom.to_shapely() for geom in self.geocolumn]
        return gpGeoSeries(output, index=index)

    def to_pandas(self):
        """
        Treats to_pandas and to_geopandas as the same call, which improves
        compatibility with pandas.
        """
        return self.to_geopandas()
