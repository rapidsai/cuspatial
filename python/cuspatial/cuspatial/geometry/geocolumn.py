# Copyright (c) 2021-2022 NVIDIA CORPORATION
import numbers
from functools import cached_property
from collections.abc import Iterable
from typing import Tuple, TypeVar

import pyarrow as pa
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cudf
from cudf.core.column import NumericalColumn

from cuspatial.geometry.geometa import GeoMeta

T = TypeVar("T", bound="GeoColumn")


class GeoColumn(NumericalColumn):
    """
    Parameters
    ----------
    data : A tuple of four cudf.ListSeries
    meta : A GeoMeta object (optional)

    Notes
    -----
    The GeoColumn class subclasses `NumericalColumn`. Combined with
    `_copy_type_metadata`, this assures support for sort, groupby,
    and potential other `cudf` algorithms.
    """

    def __init__(
        self,
        data: Tuple,
        meta: GeoMeta = None,
        shuffle_order: cudf.Index = None,
    ):
        if (
            not isinstance(data[0], cudf.Series)
            or not isinstance(data[1], cudf.Series)
            or not isinstance(data[2], cudf.Series)
            or not isinstance(data[3], cudf.Series)
        ):
            raise TypeError("All Tuple arguments must be cudf.ListSeries")
        if meta is not None:
            self._meta = meta
        else:
            self._meta = GeoMeta(data)
        self.points = data[0]
        self.points.name = "points"
        self.mpoints = data[1]
        self.mpoints.name = "mpoints"
        self.lines = data[2]
        self.lines.name = "lines"
        self.polygons = data[3]
        self.polygons.name = "polygons"
        base = cudf.core.column.column.arange(0, len(self), dtype="int64").data
        super().__init__(base, dtype="int64")
        if shuffle_order is not None:
            self._data = shuffle_order

    def to_arrow(self):
        return pa.UnionArray.from_dense(
            self._meta.type_codes.to_arrow(),
            self._meta.union_offsets.to_arrow(),
            (
                self.points.to_arrow(),
                self.mpoints.to_arrow(),
                self.lines.to_arrow(),
                self.polygons.to_arrow(),
            ),
        )

    def __getitem__(self, item):
        """
        Returns ShapelySerializer objects for each of the rows specified by
        index.
        """
        if not isinstance(item, numbers.Integral) and not isinstance(
            item, slice
        ):
            raise NotImplementedError
        # Map Step
        index = self._data[item] if self._data is not None else item
        return self.iloc[index]

    @property
    def loc(self):
        """
        Not currently supported.
        """
        return GeoColumnLocIndexer(self)

    @property
    def iloc(self):
        """
        Return the i-th row of the GeoSeries.
        """
        return GeoColumnILocIndexer(self)

    def __len__(self):
        """
        Returns the number of unique geometries stored in this GeoColumn.
        """
        return len(self._meta.input_types)

    def _dump(self):
        return (
            f"POINTS\n"
            f"{self.points._repr__()}\n"
            f"MULTIPOINTS\n"
            f"{self.multipoints._repr__()}\n"
            f"LINES\n"
            f"{self.lines._repr__()}\n"
            f"POLYGONS\n"
            f"{self.polygons._repr__()}\n"
        )

    def __repr__(self):
        return (
            f"GeoColumn\n"
            f"{len(self.points)} POINTS\n"
            f"{len(self.mpoints)} MULTIPOINTS\n"
            f"{len(self.lines)} LINES\n"
            f"{len(self.polygons)} POLYGONS\n"
        )

    def copy(self, deep=True):
        """TODO"""
        """
        Create a copy of all of the GPU-backed data structures in this
        GeoColumn.
        """
        result = GeoColumn(
            (
                self.points.copy(deep),
                self.mpoints.copy(deep),
                self.lines.copy(deep),
                self.polygons.copy(deep),
            ),
            self._meta.copy(),
            self.data.copy(),
        )
        return result


class GeoColumnLocIndexer:
    """
    Not yet supported.
    """

    def __init__(self):
        # Todo: Easy to implement with a join.
        raise NotImplementedError


from cuspatial.io.geopandas_reader import Feature_Enum


class GeoColumnILocIndexer:

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
        from cuspatial.io.geopandas_reader import Feature_Enum

        if not isinstance(index, numbers.Integral) and not isinstance(
            index, slice
        ):
            raise NotImplementedError(
                "Can't GeoColumn indexing only supports int and slice(int)"
                " at this time"
            )

        # Fix types: There's only four fields
        result_types = self._sr._meta.input_types.to_arrow()
        union_types = self._sr._meta.input_types.replace(3, 2)
        union_types = union_types.replace(4, 3)
        union_types = union_types.replace(5, 3).values_host
        result_indexes = self._sr._meta.union_offsets
        shapely_classes = [
            self._get_shapely_class_for_Feature_Enum[Feature_Enum(x)]
            for x in result_types.to_numpy()
        ]

        union = pa.UnionArray.from_dense(
            pa.array(union_types),
            self._sr._meta.union_offsets.to_arrow(),
            [
                self._sr.points.to_arrow(),
                self._sr.mpoints.to_arrow(),
                self._sr.lines.to_arrow(),
                self._sr.polygons.to_arrow(),
            ],
        )

        if isinstance(index, Iterable):
            utypes = union_types[index]
            indexes = index
            classes = shapely_classes[index]
        else:
            utypes = [union_types[index]]
            indexes = [index]
            classes = [shapely_classes[index]]

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
                            LineString([tuple(child) for child in linestring])
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
