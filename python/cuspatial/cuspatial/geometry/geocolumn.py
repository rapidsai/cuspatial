# Copyright (c) 2021 NVIDIA CORPORATION
import numbers
from itertools import repeat
from typing import TypeVar, Union

import numpy as np
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
            self.input_types.extend(repeat("p", len(buffers.points)))
            self.input_lengths.extend(repeat(1, len(buffers.points)))
        if buffers.multipoints is not None:
            self.input_types.extend(repeat("mp", len(buffers.multipoints)))
            self.input_lengths.extend(repeat(1, len(buffers.multipoints)))
        if buffers.lines is not None:
            if len(buffers.lines.mlines) > 0:
                self.input_types.extend(repeat("l", buffers.lines.mlines[0]))
                self.input_lengths.extend(repeat(1, buffers.lines.mlines[0]))
                for ml_index in range(len(buffers.lines.mlines) // 2):
                    self.input_types.extend(["ml"])
                    self.input_lengths += [1]
                    mline_size = (
                        buffers.lines.mlines[ml_index * 2 + 1]
                        - 1
                        - buffers.lines.mlines[ml_index * 2]
                    )
                    self.input_types.extend(repeat("l", mline_size))
                    self.input_lengths.extend(repeat(1, mline_size))
            else:
                self.input_types.extend(repeat("l", len(buffers.lines)))
                self.input_lengths.extend(repeat(1, len(buffers.lines)))
        if buffers.polygons is not None:
            if len(buffers.polygons.mpolys) > 0:
                self.input_types.extend(
                    repeat("poly", buffers.polygons.mpolys[0])
                )
                self.input_lengths.extend(
                    repeat(1, buffers.polygons.mpolys[0])
                )
                for mp_index in range(len(buffers.polygons.mpolys) // 2):
                    mpoly_size = (
                        buffers.polygons.mpolys[mp_index * 2 + 1]
                        - buffers.polygons.mpolys[mp_index * 2]
                    )
                    self.input_types.extend(["mpoly"])
                    self.input_lengths.extend([mpoly_size])
                    self.input_types.extend(repeat("poly", mpoly_size))
                    self.input_lengths.extend(repeat(1, mpoly_size))
            else:
                self.input_types.extend(repeat("poly", len(buffers.polygons)))
                self.input_lengths.extend(repeat(1, len(buffers.polygons)))

    def copy(self):
        return type(self)(
            {
                "input_types": self.input_types.copy(),
                "input_lengths": self.input_lengths.copy(),
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
        result = GeoColumn(
            self._geo.copy(deep), self._meta.copy(), self.data.copy()
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
            "p": PointShapelySerializer,
            "mp": MultiPointShapelySerializer,
            "l": LineStringShapelySerializer,
            "ml": MultiLineStringShapelySerializer,
            "poly": PolygonShapelySerializer,
            "mpoly": MultiPolygonShapelySerializer,
        }
        return type_map[self._sr._meta.input_types[index]](self._sr, index)


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
        return MultiPoint(np.array(result).reshape(item_length // 2, 2))


class LineStringShapelySerializer(ShapelySerializer):
    def to_shapely(self):
        """
        Finds the start and end position in the GeoArrow array of lines
        of the LineString referenced by `self._index`, creates one, and
        returns it.
        """
        ml_index = self._index - 1
        preceding_line_count = 0
        preceding_ml_count = 0
        # Skip over any LineStrings that are part of a MultiLineString
        while ml_index >= 0:
            if self._source._meta.input_types[ml_index] == "ml":
                preceding_ml_count = preceding_ml_count + 1
            elif (
                self._source._meta.input_types[ml_index] == "l"
                and preceding_ml_count == 0
            ):
                preceding_line_count = preceding_line_count + 1
            ml_index = ml_index - 1
        preceding_multis = preceding_ml_count
        if preceding_multis > 0:
            multi_end = self._source.lines.mlines[preceding_multis * 2 - 1]
            item_start = multi_end + preceding_line_count
        else:
            item_start = preceding_line_count
        item_length = self._source._meta.input_lengths[self._index]
        item_end = item_length + item_start
        item_source = self._source.lines
        result = item_source[item_start:item_end]
        return LineString(
            np.array(result).reshape(2 * (item_start - item_end), 2)
        )


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
            if self._source._meta.input_types[i] == item_type:
                index = index + 1
        line_indices = slice(
            self._source.lines.mlines[index * 2],
            self._source.lines.mlines[index * 2 + 1],
        )
        return MultiLineString(
            [
                LineString(
                    np.array(self._source.lines[i]).reshape(
                        int(len(self._source.lines[i]) / 2), 2
                    )
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
        mp_index = self._index - 1
        preceding_poly_count = 0
        preceding_mp_count = 0
        while mp_index >= 0:
            if self._source._meta.input_types[mp_index] == "mpoly":
                preceding_mp_count = preceding_mp_count + 1
            elif (
                self._source._meta.input_types[mp_index] == "poly"
                and preceding_mp_count == 0
            ):
                preceding_poly_count = preceding_poly_count + 1
            mp_index = mp_index - 1
        preceding_multis = preceding_mp_count
        multi_index = (
            self._source.polygons.mpolys[preceding_multis * 2 - 1]
            if preceding_multis > 0
            else 0
        )
        preceding_polys = preceding_poly_count
        ring_start = self._source.polygons.polys[multi_index + preceding_polys]
        ring_end = self._source.polygons.polys[
            multi_index + preceding_polys + 1
        ]
        rings = self._source.polygons.rings
        exterior_slice = slice(rings[ring_start], rings[ring_start + 1])
        exterior = self._source.polygons.xy[exterior_slice]
        return Polygon(
            np.array(exterior).reshape(2 * (ring_start - ring_end), 2),
            [
                np.array(self._source.polygons.xy[interior_slice]).reshape(
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
        item_type = self._source._meta.input_types[self._index]
        index = 0
        for i in range(self._index):
            if self._source._meta.input_types[i] == item_type:
                index = index + 1
        poly_indices = slice(
            self._source.polygons.mpolys[index * 2],
            self._source.polygons.mpolys[index * 2 + 1],
        )
        polys = []
        for i in range(poly_indices.start, poly_indices.stop):
            ring_start = self._source.polygons.polys[i]
            ring_end = self._source.polygons.polys[i + 1]
            rings = self._source.polygons.rings
            exterior_slice = slice(rings[ring_start], rings[ring_start + 1])
            exterior = self._source.polygons.xy[exterior_slice]
            polys.append(
                Polygon(
                    np.array(exterior).reshape(2 * (ring_start - ring_end), 2),
                    [
                        np.array(
                            self._source.polygons.xy[interior_slice]
                        ).reshape(
                            int(
                                (
                                    interior_slice.stop
                                    - interior_slice.start
                                    + 1
                                )
                                / 2
                            ),
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
