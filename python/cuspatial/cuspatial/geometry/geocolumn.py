# Copyright (c) 2021-2022 NVIDIA CORPORATION
import numbers
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
        self.mpoints = data[1]
        self.lines = data[2]
        self.polygons = data[3]
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
        if not isinstance(item, numbers.Integral):
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


class GeoColumnILocIndexer:
    """
    Each row of a GeoSeries is one of the six types: Point, MultiPoint,
    LineString, MultiLineString, Polygon, or MultiPolygon.
    """

    def __init__(self, sr):
        self._sr = sr

    def type_int_to_field(self, type_int):
        from cuspatial.io.geopandas_reader import Feature_Enum

        return {
            Feature_Enum.POINT: self._sr.points,
            Feature_Enum.MULTIPOINT: self._sr.mpoints,
            Feature_Enum.LINESTRING: self._sr.lines,
            Feature_Enum.MULTILINESTRING: self._sr.lines,
            Feature_Enum.POLYGON: self._sr.polygons,
            Feature_Enum.MULTIPOLYGON: self._sr.polygons,
        }[Feature_Enum(type_int)]

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
            raise NotImplementedError(
                "Can't index GeoSeries with non-integer at this time"
            )
        result_types = self._sr._meta.input_types[index]
        field = self.type_int_to_field(result_types)
        result_index = self._sr._meta.union_offsets[index]
        shapely_class = self._getitem_int(result_types)
        if result_types == 0:
            result = field[result_index]
            return shapely_class(result)
        if result_types == 1:
            points = field[result_index]
            return shapely_class(points)
        if result_types == 2:
            linestring = field[result_index]
            result = [tuple(x) for x in linestring[0]]
            return shapely_class(result)
        if result_types == 3:
            linestrings = []
            for linestring in field[result_index]:
                linestrings.append(
                    LineString([tuple(child) for child in linestring])
                )
            return shapely_class(linestrings)
        if result_types == 4:
            rings = []
            for ring in field[result_index][0]:
                rings.append(tuple(tuple(point) for point in ring))
            return shapely_class(rings[0], rings[1:])
        if result_types == 5:
            polygons = []
            for p in field[result_index]:
                rings = []
                for ring in p:
                    rings.append(tuple([tuple(point) for point in ring]))
                polygons.append(Polygon(rings[0], rings[1:]))
            return shapely_class(polygons)

    def _getitem_int(self, index):
        type_map = {
            0: Point,
            1: MultiPoint,
            2: LineString,
            3: MultiLineString,
            4: Polygon,
            5: MultiPolygon,
        }
        return type_map[index]
