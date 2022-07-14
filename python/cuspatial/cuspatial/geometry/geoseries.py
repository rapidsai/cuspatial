# Copyright (c) 2020-2022, NVIDIA CORPORATION

import numbers
from typing import TypeVar, Union, List, Tuple

import geopandas as gpd
import pandas as pd
import pyarrow as pa
from geopandas.geoseries import GeoSeries as gpGeoSeries

import cudf

from cuspatial.geometry.geocolumn import GeoColumn, GeoMeta

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
        data: Union[gpd.GeoSeries, Tuple],
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
            from cuspatial.io.geopandas_reader import GeoPandasReader

            adapter = GeoPandasReader(data)
            pandas_meta = GeoMeta(adapter.get_geopandas_meta())
            column = GeoColumn(adapter._get_geotuple(), pandas_meta)
        else:
            raise TypeError(
                f"Incompatible object passed to GeoSeries ctor {type(data)}"
            )
        super().__init__(column, index, dtype, name, nan_as_null)

    @property
    def type(self):
        gpu_types = cudf.Series(self._column._meta.input_types).astype("str")
        result = gpu_types.replace(
            {
                "0": "Point",
                "1": "MultiPoint",
                "2": "Linestring",
                "3": "MultiLinestring",
                "4": "Polygon",
                "5": "MultiPolygon",
            }
        )
        return result

    class GeoColumnAccessor:
        def __init__(self, list_series):
            self._series = list_series
            self._col = self._series._column

        @property
        def x(self):
            result = cudf.Series(self._col.leaves().values[0::2])
            return result

        @property
        def y(self):
            return cudf.Series(self._col.leaves().values[1::2])
            return result

        @property
        def xy(self):
            return cudf.Series(self._col.leaves().values)
            return result

    @property
    def points(self):
        """
        Access the `PointsArray` of the underlying `GeoArrowBuffers`.
        """
        return self.GeoColumnAccessor(self._column.points)

    @property
    def multipoints(self):
        """
        Access the `MultiPointArray` of the underlying `GeoArrowBuffers`.
        """
        return self.GeoColumnAccessor(self._column.mpoints)

    @property
    def lines(self):
        """
        Access the `LineArray` of the underlying `GeoArrowBuffers`.
        """
        return self.GeoColumnAccessor(self._column.lines)

    @property
    def polygons(self):
        """
        Access the `PolygonArray` of the underlying `GeoArrowBuffers`.
        """
        return self.GeoColumnAccessor(self._column.polygons)

    def __repr__(self):
        # TODO: Implement Iloc with slices so that we can use `Series.__repr__`
        return self.to_pandas().__repr__()

    def type_int_to_field(self, type_int):
        from cuspatial.io.geopandas_reader import Feature_Enum, Field_Enum

        return {
            Feature_Enum.POINT: self.points,
            Feature_Enum.MULTIPOINT: self.mpoints,
            Feature_Enum.LINESTRING: self.lines,
            Feature_Enum.MULTILINESTRING: self.lines,
            Feature_Enum.POLYGON: self.polygons,
            Feature_Enum.MULTIPOLYGON: self.polygons,
        }[type_int]

    def __getitem__(self, index):
        """
        NOTE:
        Using GeoMeta, we're hacking together the logic for a
        UnionColumn. We don't want to implement this in cudf at
        this time.
        TODO: Do this. So far we're going to stick to one element
        at a time like in the previous implementation.
        """
        if not isinstance(index, numbers.Integral):
            raise TypeError(
                "Can't index GeoSeries with non-integer at this time"
            )
        result_types = self._column._meta.input_types[index]
        result_index = self._column._meta.input_lengths[index]
        field = self.type_int_to_field(result_index)
        return field[result_index]

    def to_geopandas(self, nullable=False):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        if nullable is True:
            raise ValueError("GeoSeries doesn't support <NA> yet")
        output = [self._column[i] for i in range(len(self._column))]
        return gpGeoSeries(output, index=self.index.to_pandas())

    def to_pandas(self):
        """
        Treats to_pandas and to_geopandas as the same call, which improves
        compatibility with pandas.
        """
        return self.to_geopandas()
