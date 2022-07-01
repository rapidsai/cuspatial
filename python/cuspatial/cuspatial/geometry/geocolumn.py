# Copyright (c) 2021-2022 NVIDIA CORPORATION
import numbers
from itertools import repeat
from typing import TypeVar, Union

import numpy as np
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

from cuspatial.geometry.geoarrowbuffers import GeoArrowBuffers

T = TypeVar("T", bound="GeoColumn")


class GeoMeta:
    """
    Creates input_types and input_lengths for GeoColumns that are created
    using native GeoArrowBuffers. These will be used to convert to GeoPandas
    GeoSeries if necessary.
    """

    def __init__(self, meta: Union[GeoArrowBuffers, dict]):
        if isinstance(meta, dict):
            self.input_types = meta["input_types"]
            self.input_lengths = meta["input_lengths"]
            return
        buffers = meta
        self.input_types = []
        self.input_lengths = []
        if buffers.points is not None:
            self.input_types.extend(repeat(0, len(buffers.points)))
            self.input_lengths.extend(repeat(1, len(buffers.points)))
        if buffers.multipoints is not None:
            self.input_types.extend(repeat(1, len(buffers.multipoints)))
            self.input_lengths.extend(repeat(1, len(buffers.multipoints)))
        if buffers.lines is not None:
            for index in range(len(buffers.lines.mlines) - 1):
                line_len = buffers.lines.mlines[index + 1] - buffers.lines.mlines[index]
                if line_len > 1:
                    self.input_types.extend([3])
                    self.input_lengths.extend([line_len])
                else:
                    self.input_types.extend([2])
                    self.input_lengths.extend([1])
        if buffers.polygons is not None:
            for index in range(len(buffers.polygons.mpolys) - 1):
                poly_len = (
                    buffers.polygons.mpolys[index + 1] - buffers.polygons.mpolys[index]
                )
                if poly_len > 1:
                    self.input_types.extend([5])
                    self.input_lengths.extend([poly_len])
                else:
                    self.input_types.extend([4])
                    self.input_lengths.extend([1])
        self.input_types = pa.array(self.input_types).cast(pa.int8())
        self.input_lengths = pa.array(self.input_lengths).cast(pa.int32())

    def copy(self):
        return type(self)(
            {
                "input_types": pa.Int8Array.from_buffers(
                    self.input_types.type,
                    len(self.input_types),
                    self.input_types.buffers(),
                ),
                "input_lengths": pa.Int32Array.from_buffers(
                    self.input_lengths.type,
                    len(self.input_lengths),
                    self.input_lengths.buffers(),
                ),
            }
        )


class GeoColumn(NumericalColumn):
    """
    Parameters
    ----------
    data : A GeoArrowBuffers object
    meta : A GeoMeta object (optional)

    Notes
    -----
    The GeoColumn class subclasses `NumericalColumn`. Combined with
    `_copy_type_metadata`, this assures support for existing cudf algorithms.
    """

    def __init__(
        self,
        data: GeoArrowBuffers,
        meta: GeoMeta = None,
        shuffle_order: cudf.Index = None,
    ):
        base = cudf.core.column.column.arange(0, len(data), dtype="int64").data
        super().__init__(base, dtype="int64")
        self._geo = data
        if meta is not None:
            self._meta = meta
        else:
            self._meta = GeoMeta(data)
        if shuffle_order is not None:
            self._data = shuffle_order

    def to_host(self):
        result = GeoColumn(self._geo.to_host(), self._meta.copy(), self.data)
        return result

    def __getitem__(self, item):
        """
        Returns ShapelySerializer objects for each of the rows specified by
        index.
        """
        if not isinstance(item, numbers.Integral):
            raise NotImplementedError
        return self.iloc[item]

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

    @property
    def points(self):
        return self._geo._points

    @property
    def multipoints(self):
        return self._geo._multipoints

    @property
    def lines(self):
        return self._geo._lines

    @property
    def polygons(self):
        return self._geo._polygons

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
            f"{len(self.multipoints)} MULTIPOINTS\n"
            f"{len(self.lines)} LINES\n"
            f"{len(self.polygons)} POLYGONS\n"
        )

    def copy(self, deep=True):
        """
        Create a copy of all of the GPU-backed data structures in this
        GeoColumn.
        """
        result = GeoColumn(self._geo.copy(deep), self._meta.copy(), self.data.copy())
        return result

    def from_arrow(self):
        """
        I know what to do!
        """
        print("Not ready to convert from arrow")
        breakpoint()


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

    def __getitem__(self, index):
        if not isinstance(index, slice):
            mapped_index = int(self._sr.values[index])
            return self._getitem_int(mapped_index)
        else:
            raise NotImplementedError
            # This slice functionality is not ready yet
            # return self._getitem_slice(index)

    def _getitem_int(self, index):
        type_map = {
            0: PointShapelySerializer,
            1: MultiPointShapelySerializer,
            2: LineStringShapelySerializer,
            3: MultiLineStringShapelySerializer,
            4: PolygonShapelySerializer,
            5: MultiPolygonShapelySerializer,
        }
        return type_map[self._sr._meta.input_types[index].as_py()](self._sr, index)


class ShapelySerializer:
    def __init__(self, source, index):
        """
        The base class of individual GPU geometries. This and its inheriting
        classes do not manage any GPU data directly - each ShapelySerializer
        simply stores a reference to the GeoSeries it is stored within and
        the index of the geometry within the GeoSeries. Child
        ShapelySerializer classes contain the logic necessary to serialize
        and convert GPU data back to Shapely.
        """
        self._source = source
        self._index = index


class PointShapelySerializer(ShapelySerializer):
    def to_shapely(self):
        """
        Finds the position in the GeoArrow array of points that corresponds
        to the row of the column stored at `self._index`.
        """
        item_type = self._source._meta.input_types[self._index]
        types = self._source._meta.input_types[0 : self._index]
        index = 0
        for i in range(self._index):
            if types[i] == item_type:
                index = index + 1
        return Point(self._source.points[index].reset_index(drop=True))


class MultiPointShapelySerializer(ShapelySerializer):
    def to_shapely(self):
        """
        Finds the position in the GeoArrow array of multipoints that
        corresponds to the row of the column stored at `self._index`. Returns
        `item_length` coordinates starting at that position.
        """
        item_type = self._source._meta.input_types[self._index]
        types = self._source._meta.input_types[0 : self._index]
        item_start = 0
        for i in range(self._index):
            if types[i] == item_type:
                item_start = item_start + 1
        item_length = (
            self._source.multipoints._offsets[item_start + 1]
            - self._source.multipoints._offsets[item_start]
        )
        item_source = self._source.multipoints
        result = item_source[item_start]
        return MultiPoint(result.to_numpy().reshape(item_length, 2))


class LineStringShapelySerializer(ShapelySerializer):
    def to_shapely(self):
        """
        Finds the start and end position in the GeoArrow array of lines
        of the LineString referenced by `self._index`, creates one, and
        returns it.
        """
        index = 0
        for i in range(self._index):
            if (
                self._source._meta.input_types[i] == pa.array([2]).cast(pa.int8())[0]
                or self._source._meta.input_types[i] == pa.array([3]).cast(pa.int8())[0]
            ):
                index = index + 1
        ring_start = self._source.lines.mlines[index]
        ring_end = self._source.lines.mlines[index + 1]
        rings = self._source.lines.offsets * 2
        item_start = rings[ring_start]
        item_end = rings[ring_end]
        result = self._source.lines.xy[item_start:item_end]
        return LineString(result.to_numpy().reshape(2 * (item_start - item_end), 2))


class MultiLineStringShapelySerializer(ShapelySerializer):
    def to_shapely(self):
        """
        Finds the range of LineStrings that are specified by the mlines values.
        Count the number of MultiLines stored prior to the one referenced by
        `self._index`, then return the MultiLineString at that position packed
        with the LineStrings in its range.
        """
        item_type = self._source._meta.input_types[self._index]
        index = 0
        for i in range(self._index):
            if (
                self._source._meta.input_types[i] == pa.array([2]).cast(pa.int8())[0]
                or self._source._meta.input_types[i] == pa.array([3]).cast(pa.int8())[0]
            ):
                index = index + 1
        line_indices = slice(
            self._source.lines.mlines[index],
            self._source.lines.mlines[index + 1],
        )
        return MultiLineString(
            [
                LineString(
                    self._source.lines[i]
                    .to_numpy()
                    .reshape(int(len(self._source.lines[i]) / 2), 2)
                )
                for i in range(line_indices.start, line_indices.stop, 1)
            ]
        )


class PolygonShapelySerializer(ShapelySerializer):
    """
    Find the polygon rings and coordinates in the self._index-th row of the
    column. Find the last index of the MultiPolygons that precede the
    desired Polygon, and the number of Polygons that fall between the last
    MultiPolygon and the desired Polygon. This identifies the index of the
    first ring of the Polygon. Construct a new Polygon using the first ring
    as exterior, and subsequent interior rings.
    """

    def to_shapely(self):
        index = 0
        for i in range(self._index):
            if (
                self._source._meta.input_types[i] == pa.array([4]).cast(pa.int8())[0]
                or self._source._meta.input_types[i] == pa.array([5]).cast(pa.int8())[0]
            ):
                index = index + 1
        polygon_start = self._source.polygons.mpolys[index]
        polygon_end = self._source.polygons.mpolys[index + 1]
        ring_start = self._source.polygons.polys[polygon_start]
        ring_end = self._source.polygons.polys[polygon_end]
        rings = self._source.polygons.rings * 2
        exterior_slice = slice(rings[ring_start], rings[ring_start + 1])
        exterior = self._source.polygons.xy[exterior_slice]
        return Polygon(
            exterior.to_numpy().reshape(2 * (ring_start - ring_end), 2),
            [
                self._source.polygons.xy[interior_slice]
                .to_numpy()
                .reshape(
                    int((interior_slice.stop - interior_slice.start + 1) / 2),
                    2,
                )
                for interior_slice in [
                    slice(rings[interior], rings[interior + 1])
                    for interior in range(ring_end - (ring_start + 1))
                    + (ring_start + 1)
                ]
            ],
        )


class MultiPolygonShapelySerializer(ShapelySerializer):
    def to_shapely(self):
        """
        Iteratively construct Polygons from exterior (first) rings and
        subsequent interior rings of all polygons that around bound by the
        mpolygon specified by self._index.
        """
        index = 0
        for i in range(self._index):
            if (
                self._source._meta.input_types[i] == pa.array([4]).cast(pa.int8())[0]
                or self._source._meta.input_types[i] == pa.array([5]).cast(pa.int8())[0]
            ):
                index = index + 1
        poly_indices = slice(
            self._source.polygons.mpolys[index],
            self._source.polygons.mpolys[index + 1],
        )
        polys = []
        for i in range(poly_indices.start, poly_indices.stop):
            ring_start = self._source.polygons.polys[i]
            ring_end = self._source.polygons.polys[i + 1]
            rings = self._source.polygons.rings * 2
            exterior_slice = slice(rings[ring_start], rings[ring_start + 1])
            exterior = self._source.polygons.xy[exterior_slice]
            polys.append(
                Polygon(
                    exterior.to_numpy().reshape(2 * (ring_start - ring_end), 2),
                    [
                        self._source.polygons.xy[interior_slice]
                        .to_numpy()
                        .reshape(
                            int((interior_slice.stop - interior_slice.start + 1) / 2),
                            2,
                        )
                        for interior_slice in [
                            slice(rings[interior], rings[interior + 1])
                            for interior in range(ring_start + 1, ring_end)
                        ]
                    ],
                )
            )
        return MultiPolygon(polys)
