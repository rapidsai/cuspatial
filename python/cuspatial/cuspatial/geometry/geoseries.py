# Copyright (c) 2020-2021, NVIDIA CORPORATION

import numbers
from geopandas.geoseries import GeoSeries as gpGeoSeries
import pandas as pd
import numpy as np

from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)

import cudf
from cudf.core.column import ColumnBase

from cuspatial.io.geoseries_reader import GeoSeriesReader


class GeoSeries(ColumnBase):
    def __init__(self, data, name=None, index=None):
        """
        A GPU GeoSeries object.

        Parameters
        ----------
        data : A GeoPandas GeoSeries object, or a file path

        Discussion
        ----------
        The GeoArrow format specifies a tabular data format for geometry
        information. Supported types include Point, MultiPoint, LineString,
        MultiLineString, Polygon, and MultiPolygon.  In order to store
        these coordinate types in a strictly tabular fashion, columns are
        created for Points, MultiPoints, LineStrings, and Polygons.
        MultiLines and MultiPolygons are stored in the same data structure
        as LineStrings and Polygons.

        The Points column is a simple numeric column. x and y coordinates
        can be stored either interleaved or in separate columns. If a z
        coordinate is present, it will be stored in a separate column.

        The MultiPoints column is similar to the Points column with the
        addition of an offsets column. The offsets column stores the comparable
        sizes and coordinates of each MultiPoint in the cuGeoSeries.

        LineStrings contain the coordinates column, an offsets column, and a
        multioffsets column. The multioffsets column stores the indices of the
        offsets that indicate the beginning and end of each MultiLineString
        segment.

        Polygons contain the coordinates column, a rings column specifying
        the beginning and end of every polygon, a polygons column specifying
        the beginning, or exterior, ring of each polygon and the end ring.
        All rings after the first ring are interior rings.  Finally a
        multipolys column stores the offsets of the polygons that should be
        grouped into MultiPolygons.

        As a result, a GpuGeoSeries object contains these arrow-supported
        columns:
        points_xy
        mpoints_xy
        mpoints_offsets
        lines_xy
        lines_offsets
        lines_ml_offsets
        polygons_xy
        polygons_rings
        polygons_polys
        polygons_mpolys

        Notes
        -----
        Legacy cuspatial algorithms depend on separated x and y columns.
        De-interleave x and y columns with `column.de_interleave`, which
        will create a legacy cuspatial Series, which violates the GeoArrow
        format but will work with existing cuspatial algorithms.
        """
        if isinstance(data, GeoSeries):
            self._data = data.data
            self._points = data.points
            self._multipoints = data.multipoints
            self._lines = data.lines
            self._polygons = data.polygons
            self.types = data.types
            self.lengths = data.lengths
            if index is not None:
                self.index = index
            else:
                self.index = data.index
            self.name = data.name if not name else name
            self._dtype = data.dtype
        else:
            self._data = data
            self._reader = GeoSeriesReader(data)
            self._points = GpuPointArray()
            self._points.xy = self._reader.buffers[0]["points"]
            self._multipoints = GpuMultiPointArray()
            self._multipoints.xy = self._reader.buffers[0]["multipoints"]
            self._multipoints.offsets = self._reader.buffers[1]["multipoints"]
            self._lines = GpuLineArray()
            self._lines.xy = self._reader.buffers[0]["lines"]
            self._lines.offsets = self._reader.buffers[1]["lines"]
            self._lines.mlines = self._reader.buffers[1]["mlines"]
            self._polygons = GpuPolygonArray()
            self._polygons.xy = self._reader.buffers[0]["polygons"]["coords"]
            self._polygons.polys = cudf.Series(
                self._reader.offsets["polygons"]["polygons"]
            )
            self._polygons.rings = cudf.Series(
                self._reader.offsets["polygons"]["rings"]
            )
            self._polygons.mpolys = cudf.Series(
                self._reader.offsets["polygons"]["mpolys"]
            )
            self.types = self._reader.buffers[2]
            self.lengths = self._reader.buffers[3]
            if index is not None:
                self.index = index
            elif data.index is not None:
                self.index = data.index
            else:
                self.index = cudf.Series(np.arange(len(self)))
            self.name = name
            self._dtype = "geometry"

    class GeoSeriesLocIndexer:
        def __init__(self):
            raise NotImplementedError

    class GeoSeriesILocIndexer:
        def __init__(self, sr):
            self._sr = sr

        def __getitem__(self, index):
            if not isinstance(index, slice):
                return self._getitem_int(index)
            else:
                raise NotImplementedError
                # This slice functionality is not ready yet
                # return self._getitem_slice(index)

        def _getitem_int(self, index):
            item_type = self._sr.types[index]
            if item_type == "p":
                return gpuPoint(self._sr, index)
            elif item_type == "mp":
                return gpuMultiPoint(self._sr, index)
            elif item_type == "l":
                return gpuLineString(self._sr, index)
            elif item_type == "ml":
                return gpuMultiLineString(self._sr, index)
            elif item_type == "poly":
                return gpuPolygon(self._sr, index)
            elif item_type == "mpoly":
                return gpuMultiPolygon(self._sr, index)
            else:
                raise TypeError

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
        return self.GeoSeriesLocIndexer(self)

    @property
    def iloc(self):
        return self.GeoSeriesILocIndexer(self)

    def __len__(self):
        """
        Returns the number of unique geometries stored in this cuGeoSeries.
        """
        return int(len(self.types))

    @property
    def points(self):
        return self._points

    @property
    def multipoints(self):
        return self._multipoints

    @property
    def lines(self):
        return self._lines

    @property
    def polygons(self):
        return self._polygons

    def to_pandas(self, index=None, nullable=False):
        return self.to_geopandas(index=index, nullable=nullable)

    def to_geopandas(self, index=None, nullable=False):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        output = []
        if nullable is True:
            raise ValueError("cuGeoSeries doesn't support N/A yet")
        if index is None:
            index = self.index
            if isinstance(index, cudf.core.index.RangeIndex):
                index = index.to_pandas()
        output = [geom.to_shapely() for geom in self]
        return gpGeoSeries(output, index=index)

    def __repr__(self):
        return self.to_pandas().__repr__() + "\n" + "(GPU)" + "\n"

    def _dump(self):
        return (
            "POINTS"
            + "\n"
            + self.points.__repr__()
            + "MULTIPOINTS"
            + "\n"
            + self.multipoints.__repr__()
            + "LINES"
            + "\n"
            + self.lines.__repr__()
            + "POLYGONS"
            + "\n"
            + self.polygons.__repr__()
            + "\n"
        )

    def de_interleave(self):
        return NotImplementedError

    def copy(self, deep=True):
        result = GeoSeries([])
        result._data = self._data
        result._reader = self._reader
        result._points = self.points.copy(deep)
        result._multipoints = self.multipoints.copy(deep)
        result._lines = self.lines.copy(deep)
        result._polygons = self.polygons.copy(deep)
        result.types = self.types
        result.lengths = self.lengths
        result.index = self.index
        return result

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= len(self):
            raise StopIteration
        result = self[self._iter_index]
        self._iter_index = self._iter_index + 1
        return result


class GpuPointArray:
    def __init__(self):
        self.xy = None
        self.z = None
        self.has_z = False

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.xy):
            raise StopIteration
        result = self[self._index]
        self._index = self._index + 2
        return result

    def __getitem__(self, index):
        if isinstance(index, slice):
            new_index = slice(index.start * 2, index.stop * 2 + 1, index.step)
            return self.xy.iloc[new_index]
        return self.xy.iloc[(index * 2) : (index * 2) + 2]

    def __repr__(self):
        return "xy: " + self.xy.__repr__()

    def copy(self, deep=True):
        result = GpuPointArray()
        if self.xy is not None:
            result.xy = self.xy.copy(deep)
        if self.z is not None:
            result.z = self.z.copy(deep)
        result.has_z = self.has_z
        return result


class GpuOffsetArray(GpuPointArray):
    def __init__(self):
        super().__init__()
        self.offsets = []

    def __iter__(self):
        if self.offsets is None:
            raise TypeError
        self._index = 1
        return self

    def __next__(self):
        if self._index >= len(self.offsets):
            raise StopIteration
        result = self[self._index]
        self._index = self._index + 1
        return result

    def __getitem__(self, index):
        if isinstance(index, slice):
            rindex = index
        else:
            rindex = slice(index, index + 1, 1)
        new_slice = slice(self.offsets[rindex.start], None)
        if rindex.stop < len(self.offsets):
            new_slice = slice(new_slice.start, self.offsets[rindex.stop])
        result = self.xy[new_slice]
        return result

    def __repr__(self):
        return (
            super().__repr__() + "\noffsets: " + self.offsets.__repr__() + "\n"
        )

    def copy(self, deep=True):
        result = super().copy(deep)
        result.offsets = self.offsets.copy(deep)
        return result


class GpuLineArray(GpuOffsetArray):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        result = super().__getitem__(index)
        return result

    def __repr__(self):
        return (
            super().__repr__() + "\nmlines: " + self.mlines.__repr__() + "\n"
        )

    def copy(self, deep=True):
        result = GpuLineArray()
        base = super().copy(deep)
        result.xy = base.xy
        result.z = base.z
        result.offsets = base.offsets
        if hasattr(self, "mlines"):
            result.mlines = self.mlines.copy()
        return result


class GpuMultiPointArray(GpuOffsetArray):
    def __init__(self):
        super().__init__()

    def copy(self, deep=True):
        result = GpuMultiPointArray()
        base = super().copy(deep)
        result.xy = base.xy
        result.z = base.z
        result.offsets = base.offsets
        return result


class GpuPolygonArray(GpuOffsetArray):
    def __init__(self):
        super().__init__()
        self.polys = None
        # GpuPolygonArray uses the offsets buffer for rings!
        self.mpolys = None

    @property
    def rings(self):
        return self.offsets

    @rings.setter
    def rings(self, rings):
        self.offsets = rings

    def __repr__(self):
        result = ""
        result += "xy:\n" + self.xy.__repr__() + "\n"
        result += "polys:\n" + self.polys.__repr__() + "\n"
        result += "rings:\n" + self.rings.__repr__() + "\n"
        result += "mpolys:\n" + self.mpolys.__repr__() + "\n"
        return result

    def copy(self, deep=True):
        result = super().copy(deep)
        result.polys = self.polys.copy(deep)
        result.rings = self.rings.copy(deep)
        result.mpolys = self.mpolys.copy(deep)
        return result


class gpuGeometry:
    def __init__(self, source, index):
        self.source = source
        self.index = index

    def __repr__(self):
        return self.source.__repr__()


class gpuPoint(gpuGeometry):
    def to_shapely(self):
        item_type = self.source.types[self.index]
        index = (
            pd.Series(self.source.types[0 : self.index], dtype="object")
            == item_type
        ).sum()
        return Point(self.source._points[index].reset_index(drop=True))

    def __iter__(self):
        self._index = 0
        self._shapely = self.to_shapely()
        return self

    def __next__(self):
        if self._index >= 1:
            raise StopIteration
        result = self._shapely
        self._index = self._index + 1
        return result


class gpuMultiPoint(gpuGeometry):
    def to_shapely(self):
        item_type = self.source.types[self.index]
        item_length = self.source.lengths[self.index]
        item_start = (
            pd.Series(self.source.types[0 : self.index], dtype="object")
            == item_type
        ).sum()
        item_source = self.source._multipoints
        result = item_source[item_start]
        return MultiPoint(result.to_array().reshape(item_length, 2))


class gpuLineString(gpuGeometry):
    def to_shapely(self):
        ml_index = self.index - 1
        preceding_line_count = 0
        preceding_ml_count = 0
        while ml_index >= 0:
            if self.source.types[ml_index] == "ml":
                preceding_ml_count = preceding_ml_count + 1
            elif (
                self.source.types[ml_index] == "l" and preceding_ml_count == 0
            ):
                preceding_line_count = preceding_line_count + 1
            ml_index = ml_index - 1
        preceding_multis = preceding_ml_count
        if preceding_multis > 0:
            multi_end = self.source._lines.mlines[preceding_multis * 2 - 1]
            item_start = multi_end + preceding_line_count
        else:
            item_start = preceding_line_count
        item_length = self.source.lengths[self.index]
        item_end = item_length + item_start
        item_source = self.source._lines
        result = item_source[item_start:item_end]
        return LineString(
            result.to_array().reshape(2 * (item_start - item_end), 2)
        )


class gpuMultiLineString(gpuGeometry):
    def to_shapely(self):
        item_type = self.source.types[self.index]
        index = (
            pd.Series(self.source.types[0 : self.index], dtype="object")
            == item_type
        ).sum()
        line_indices = slice(
            self.source._lines.mlines[index * 2],
            self.source._lines.mlines[index * 2 + 1],
        )
        return MultiLineString(
            [
                LineString(
                    self.source._lines[i]
                    .to_array()
                    .reshape(int(len(self.source._lines[i]) / 2), 2)
                )
                for i in range(line_indices.start, line_indices.stop, 1)
            ]
        )


class gpuPolygon(gpuGeometry):
    def to_shapely(self):
        mp_index = self.index - 1
        preceding_poly_count = 0
        preceding_mp_count = 0
        while mp_index >= 0:
            if self.source.types[mp_index] == "mpoly":
                preceding_mp_count = preceding_mp_count + 1
            elif (
                self.source.types[mp_index] == "poly"
                and preceding_mp_count == 0
            ):
                preceding_poly_count = preceding_poly_count + 1
            mp_index = mp_index - 1
        preceding_multis = preceding_mp_count
        multi_index = (
            self.source._polygons.mpolys[preceding_multis * 2 - 1]
            if preceding_multis > 0
            else 0
        )
        preceding_polys = preceding_poly_count
        ring_start = self.source._polygons.polys[multi_index + preceding_polys]
        ring_end = self.source._polygons.polys[
            multi_index + preceding_polys + 1
        ]
        rings = self.source._polygons.rings
        exterior_slice = slice(rings[ring_start], rings[ring_start + 1])
        exterior = self.source._polygons.xy[exterior_slice]
        return Polygon(
            exterior.to_array().reshape(2 * (ring_start - ring_end), 2),
            [
                self.source._polygons.xy[interior_slice]
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
        item_type = self.source.types[self.index]
        index = (
            pd.Series(self.source.types[0 : self.index], dtype="object")
            == item_type
        ).sum()
        poly_indices = slice(
            self.source.polygons.mpolys[index * 2],
            self.source.polygons.mpolys[index * 2 + 1],
        )
        polys = []
        for i in range(poly_indices.start, poly_indices.stop):
            ring_start = self.source._polygons.polys[i]
            ring_end = self.source._polygons.polys[i + 1]
            rings = self.source._polygons.rings
            exterior_slice = slice(rings[ring_start], rings[ring_start + 1])
            exterior = self.source._polygons.xy[exterior_slice]
            polys.append(
                Polygon(
                    exterior.to_array().reshape(
                        2 * (ring_start - ring_end), 2
                    ),
                    [
                        self.source._polygons.xy[interior_slice]
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
