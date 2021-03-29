# Copyright (c) 2021 NVIDIA CORPORATION

import numbers
import numpy as np

from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)
from typing import TypeVar

import cudf
from cudf.core.column import ColumnBase, NumericalColumn

from cuspatial.geometry.geoarrowbuffers import GeoArrowBuffers


T = TypeVar("T", bound="GeoColumn")


class GeoMeta:
    def __init__(self, buffers: GeoArrowBuffers):
        self.input_types = []
        self.input_lengths = []
        if hasattr(buffers, "points"):
            self.input_types += list(np.repeat("p", len(buffers.points)))
            self.input_lengths += list(np.repeat(1, len(buffers.points)))
        if hasattr(buffers, "multipoints"):
            self.input_types += list(np.repeat("mp", len(buffers.multipoints)))
            self.input_lengths += list(np.repeat(1, len(buffers.multipoints)))
        if hasattr(buffers, "lines"):
            if hasattr(buffers.lines, "mlines"):
                self.input_types += list(
                    np.repeat("l", buffers.lines.mlines[0])
                )
                self.input_lengths += list(
                    np.repeat(1, buffers.lines.mlines[0])
                )
                for ml_index in range(len(buffers.lines.mlines) // 2):
                    self.input_types += list(["ml"])
                    self.input_lengths += [1]
                    self.input_types += list(
                        np.repeat(
                            "l",
                            buffers.lines.mlines[ml_index * 2 + 1]
                            - 1
                            - buffers.lines.mlines[ml_index * 2],
                        )
                    )
                    self.input_lengths += list(
                        np.repeat(
                            1,
                            buffers.lines.mlines[ml_index * 2 + 1]
                            - 1
                            - buffers.lines.mlines[ml_index * 2],
                        )
                    )
            else:
                self.input_types += list(np.repeat("l", len(buffers.lines)))
                self.input_lengths += list(np.repeat("l", len(buffers.lines)))

    def copy(self, deep=True):
        return type(self)(
            {
                "input_types": self.input_types.copy(),
                "input_lengths": self.input_lengths.copy(),
                "inputs": self.inputs.copy(),
            }
        )


class GeoPandasMeta(GeoMeta):
    """
    When a GeoColumn is created from a GeoPandas GeoSeries, this meta data
    remembers what order the geometries began in.
    """

    def __init__(self, meta: dict):
        self.input_types = meta["input_types"]
        self.input_lengths = meta["input_lengths"]
        self.inputs = meta["inputs"]


class GeoColumn(NumericalColumn):
    """
    Parameters
    ----------
    data : A GeoArrowBuffers object
    meta : A GeoPandasMeta object (optional)

    Notes
    -----
    The GeoColumn class subclasses `NumericalColumn`. Combined with
    `_copy_type_metadata`, this assures support for existing cudf algorithms.
    """

    def __init__(self, data: GeoArrowBuffers, meta: GeoPandasMeta = None):
        base = cudf.Series(cudf.RangeIndex(0, len(data)))._column.data
        super().__init__(base, dtype="int64")
        self._geo = data
        if meta is not None:
            self._meta = meta
        else:
            self._meta = GeoMeta(data)

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= len(self):
            raise StopIteration
        result = self[self._iter_index]
        self._iter_index = self._iter_index + 1
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
                "p": gpuPoint,
                "mp": gpuMultiPoint,
                "l": gpuLineString,
                "ml": gpuMultiLineString,
                "poly": gpuPolygon,
                "mpoly": gpuMultiPolygon,
            }
            return type_map[self._sr._types[index]](self._sr, index)

    def __getitem__(self, item):
        """
        Returns gpuGeometry objects for each of the rows specified by index.
        """
        if not isinstance(item, numbers.Integral):
            raise NotImplementedError
        return self.iloc[item]

    @property
    def loc(self):
        """
        Not currently supported.
        """
        return self.GeoColumnLocIndexer(self)

    @property
    def iloc(self):
        """
        Return the i-th row of the GeoSeries.
        """
        return self.GeoColumnILocIndexer(self)

    @property
    def _types(self):
        """
        A list of string types from the set "p", "mp", "l", "ml", "poly", and
        "mpoly". These are the types of each row of the GeoColumn. This
        property only exists when a GeoColumn has been created from a GeoPandas
        object.
        """
        return self._meta.input_types

    @_types.setter
    def _types(self, types):
        raise TypeError("GeoPandasMeta does not support item assignment.")

    @property
    def _lengths(self):
        """
        A list of integers of the length of each Multi geometry. Each non-multi
        geometry is length 1.
        """
        return self._meta.input_lengths

    @_lengths.setter
    def _lengths(self, lengths):
        raise TypeError("GeoPandasMeta does not support item assignment.")

    def __len__(self):
        """
        Returns the number of unique geometries stored in this GeoColumn.
        """
        if self._meta is not None:
            return len(self._meta.input_types)
        else:
            return len(self._geo)

    @property
    def points(self):
        """
        The Points column is a simple numeric column. x and y coordinates
        can be stored either interleaved or in separate columns. If a z
        coordinate is present, it will be stored in a separate column.
        """
        return self._geo._points

    @property
    def multipoints(self):
        """
        The MultiPoints column is similar to the Points column with the
        addition of an offsets column. The offsets column stores the comparable
        sizes and coordinates of each MultiPoint in the cuGeoSeries.
        """
        return self._geo._multipoints

    @property
    def lines(self):
        """
        LineStrings contain the coordinates column, an offsets column, and a
        multioffsets column. The multioffsets column stores the indices of the
        offsets that indicate the beginning and end of each MultiLineString
        segment.
        """
        return self._geo._lines

    @property
    def polygons(self):
        """
        Polygons contain the coordinates column, a rings olumn specifying
        the beginning and end of every polygon, a polygons column specifying
        the beginning, or exterior, ring of each polygon and the end ring.
        All rings after the first ring are interior rings.  Finally a
        multipolys column stores the offsets of the polygons that should be
        grouped into MultiPolygons.
        """
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

    def copy(self, deep=True):
        """
        Create a copy of all of the GPU-backed data structures in this
        GeoColumn.
        """
        result = GeoColumn(self._geo.copy(), self._meta.copy())
        return result

    def _copy_type_metadata(self: T, other: ColumnBase) -> ColumnBase:
        self._data = other.data
        return self


class gpuGeometry:
    def __init__(self, source, index):
        """
        The base class of individual GPU geometries. This and its inheriting
        classes do not manage any GPU data directly - each gpuGeometry simply
        stores a reference to the GeoSeries it is stored within and the index
        of the geometry within the GeoSeries. Child gpuGeometry classes
        contain the logic necessary to serialize and convert GPU data back to
        Shapely.
        """
        self._source = source
        self._index = index


class gpuPoint(gpuGeometry):
    def to_shapely(self):
        item_type = self._source._types[self._index]
        types = self._source._types[0 : self._index]
        index = 0
        for i in range(self._index):
            if types[i] == item_type:
                index = index + 1
        return Point(self._source.points[index].reset_index(drop=True))


class gpuMultiPoint(gpuGeometry):
    def to_shapely(self):
        item_type = self._source._types[self._index]
        types = self._source._types[0 : self._index]
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
        return MultiPoint(result.to_array().reshape(item_length // 2, 2))


class gpuLineString(gpuGeometry):
    def to_shapely(self):
        ml_index = self._index - 1
        preceding_line_count = 0
        preceding_ml_count = 0
        while ml_index >= 0:
            if self._source._types[ml_index] == "ml":
                preceding_ml_count = preceding_ml_count + 1
            elif (
                self._source._types[ml_index] == "l"
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
        item_length = self._source._lengths[self._index]
        item_end = item_length + item_start
        item_source = self._source.lines
        result = item_source[item_start:item_end]
        return LineString(
            result.to_array().reshape(2 * (item_start - item_end), 2)
        )


class gpuMultiLineString(gpuGeometry):
    def to_shapely(self):
        item_type = self._source._types[self._index]
        index = 0
        for i in range(self._index):
            if self._source._types[i] == item_type:
                index = index + 1
        line_indices = slice(
            self._source.lines.mlines[index * 2],
            self._source.lines.mlines[index * 2 + 1],
        )
        return MultiLineString(
            [
                LineString(
                    self._source.lines[i]
                    .to_array()
                    .reshape(int(len(self._source.lines[i]) / 2), 2)
                )
                for i in range(line_indices.start, line_indices.stop, 1)
            ]
        )


class gpuPolygon(gpuGeometry):
    def to_shapely(self):
        mp_index = self._index - 1
        preceding_poly_count = 0
        preceding_mp_count = 0
        while mp_index >= 0:
            if self._source._types[mp_index] == "mpoly":
                preceding_mp_count = preceding_mp_count + 1
            elif (
                self._source._types[mp_index] == "poly"
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
            exterior.to_array().reshape(2 * (ring_start - ring_end), 2),
            [
                self._source.polygons.xy[interior_slice]
                .to_array()
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


class gpuMultiPolygon(gpuGeometry):
    def to_shapely(self):
        item_type = self._source._types[self._index]
        index = 0
        for i in range(self._index):
            if self._source._types[i] == item_type:
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
                    exterior.to_array().reshape(
                        2 * (ring_start - ring_end), 2
                    ),
                    [
                        self._source.polygons.xy[interior_slice]
                        .to_array()
                        .reshape(
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
