# Copyright (c) 2020-2022, NVIDIA CORPORATION

from collections.abc import Iterable
from functools import cached_property
from typing import Tuple, TypeVar, Union

import geopandas as gpd
import pandas as pd
import pyarrow as pa
from geopandas.geoseries import GeoSeries as gpGeoSeries
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cudf

from cuspatial.geometry.geocolumn import GeoColumn, GeoMeta
from cuspatial.io.geopandas_reader import Feature_Enum

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
            return cudf.Series(self._col.leaves().values[0::2])

        @property
        def y(self):
            return cudf.Series(self._col.leaves().values[1::2])

        @property
        def xy(self):
            return cudf.Series(self._col.leaves().values)

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

    class GeoSeriesLocIndexer:
        """
        Not yet supported.
        """

        def __init__(self):
            # Todo: Easy to implement with a join.
            raise NotImplementedError

    class GeoSeriesILocIndexer:

        """
        Each row of a GeoSeries is one of the six types: Point, MultiPoint,
        LineString, MultiLineString, Polygon, or MultiPolygon.
        """

        def __init__(self, sr):
            self._sr = sr

        @cached_property
        def _type_int_to_field(self):
            return {
                Feature_Enum.POINT: self._sr.points,
                Feature_Enum.MULTIPOINT: self._sr.mpoints,
                Feature_Enum.LINESTRING: self._sr.lines,
                Feature_Enum.MULTILINESTRING: self._sr.lines,
                Feature_Enum.POLYGON: self._sr.polygons,
                Feature_Enum.MULTIPOLYGON: self._sr.polygons,
            }

        @cached_property
        def _get_shapely_class_for_Feature_Enum(self):
            type_map = {
                Feature_Enum.POINT: Point,
                Feature_Enum.MULTIPOINT: MultiPoint,
                Feature_Enum.LINESTRING: LineString,
                Feature_Enum.MULTILINESTRING: MultiLineString,
                Feature_Enum.POLYGON: Polygon,
                Feature_Enum.MULTIPOLYGON: MultiPolygon,
            }
            return type_map

        def __getitem__(self, index):
            """
            NOTE:
            Using GeoMeta, we're hacking together the logic for a
            UnionColumn. We don't want to implement this in cudf at
            this time.
            TODO: Do this. So far we're going to stick to one element
            at a time like in the previous implementation.
            """
            # Fix types: There's only four fields
            result_types = self._sr._column._meta.input_types.to_arrow()
            union_types = self._sr._column._meta.input_types.replace(3, 2)
            union_types = union_types.replace(4, 3)
            union_types = union_types.replace(5, 3).values_host
            shapely_classes = [
                self._get_shapely_class_for_Feature_Enum[Feature_Enum(x)]
                for x in result_types.to_numpy()
            ]

            union = pa.UnionArray.from_dense(
                pa.array(union_types),
                self._sr._column._meta.union_offsets.to_arrow(),
                [
                    self._sr._column.points.to_arrow(),
                    self._sr._column.mpoints.to_arrow(),
                    self._sr._column.lines.to_arrow(),
                    self._sr._column.polygons.to_arrow(),
                ],
            )

            if isinstance(index, slice):
                indexes = list(
                    range(
                        index.start if index.start else 0,
                        index.stop if index.stop else len(union),
                        index.step if index.step else 1,
                    )
                )
                utypes = union_types[index]
                classes = shapely_classes[index]
            elif isinstance(index, Iterable):
                indexes = index
                utypes = union_types[index]
                classes = shapely_classes[index]
            else:
                indexes = [index]
                utypes = [union_types[index]]
                classes = [shapely_classes[index]]
            breakpoint()
            results = []
            for result_type, result_index, shapely_class in zip(
                utypes, indexes, classes
            ):
                if result_type == 0:
                    result = union[result_index]
                    results.append(shapely_class(result.as_py()))
                elif result_type == 1:
                    points = union[result_index]
                    results.append(shapely_class(points.as_py()))
                elif result_type == 2:
                    linestring = union[result_index].as_py()
                    if len(linestring) == 1:
                        result = [tuple(x) for x in linestring[0]]
                        results.append(shapely_class(result))
                    else:
                        linestrings = []
                        for linestring in union[result_index].as_py():
                            linestrings.append(
                                LineString(
                                    [tuple(child) for child in linestring]
                                )
                            )
                        results.append(shapely_class(linestrings))
                elif result_type == 3:
                    polygon = union[result_index].as_py()
                    if len(polygon) == 1:
                        rings = []
                        for ring in polygon[0]:
                            rings.append(tuple(tuple(point) for point in ring))
                        results.append(shapely_class(rings[0], rings[1:]))
                    else:
                        polygons = []
                        for p in union[result_index].as_py():
                            rings = []
                            for ring in p:
                                rings.append(
                                    tuple([tuple(point) for point in ring])
                                )
                            polygons.append(Polygon(rings[0], rings[1:]))
                        results.append(shapely_class(polygons))

            if isinstance(index, Iterable):
                return results
            else:
                return results[0]

    @property
    def loc(self):
        """
        Not currently supported.
        """
        return self.GeoSeriesLocIndexer(self)

    @property
    def iloc(self):
        """
        Return the i-th row of the GeoSeries.
        """
        return self.GeoSeriesILocIndexer(self)

    def __getitem__(self, item):
        index = (
            self._column._data[item]
            if self._column._data is not None
            else item
        )
        return self.iloc[index]

    def to_geopandas(self, nullable=False):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        if nullable is True:
            raise ValueError("GeoSeries doesn't support <NA> yet")
        final_union_slice = self[0 : len(self)]
        return gpGeoSeries(final_union_slice, index=self.index.to_pandas())

    def to_pandas(self):
        """
        Treats to_pandas and to_geopandas as the same call, which improves
        compatibility with pandas.
        """
        return self.to_geopandas()
