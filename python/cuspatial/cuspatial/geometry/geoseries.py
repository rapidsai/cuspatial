# 2020 NVIDIA

from geopandas.geoseries import GeoSeries as gpGeoSeries
import pandas as pd

from shapely.geometry import (
    Point,
    LineString,
)

import cudf

from cuspatial.io.geoseries_reader import GeoSeriesReader


class GeoSeries:
    def __init__(self, data, interleaved=True):
        """
        A GPU GeoSeries object.

        Parameters
        ----------
        data : A GeoPandas GeoSeries object, or a file path

        interleaved : Boolean
            Store x,y coordinates in an interleaved [x,y] pattern according
            to GeoArrow form or store x and y coordinates in separate
            parallel buffers. Defaults to True.
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
        Creating a cuGeoSeries from a GeoPandas source with `interleaved=True`
        will create a legacy cuspatial Series, which violates the GeoArrow
        format but will work with existing cuspatial algorithms.
        """
        self._data = data
        self._reader = GeoSeriesReader(data, interleaved)
        self._points = GpuPoints()
        self._points.xy = self._reader.buffers[0]["points"]
        self._multipoints = GpuMultiPoints()
        self._multipoints.xy = self._reader.buffers[0]["multipoints"]
        self._multipoints.offsets = self._reader.buffers[1]["multipoints"]
        self._lines = GpuLines()
        self._lines.xy = self._reader.buffers[0]["lines"]
        self._lines.offsets = self._reader.buffers[1]["lines"]
        self._polygons = GpuPolygons()
        self._polygons.xy = self._reader.buffers[0]["polygons"]["coords"]
        self._polygons.polys = cudf.Series(
            self._reader.offsets["polygons"]["polygons"]
        )
        self._polygons.rings = cudf.Series(
            self._reader.offsets["polygons"]["rings"]
        )
        self.types = self._reader.buffers[2]
        self.lengths = self._reader.buffers[3]

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
                return self._getitem_slice(index)

        def _getitem_int(self, index):
            item_type = self._sr.types[index]
            item_length = self._sr.lengths[index]
            item_start = (
                pd.Series(self._sr.types[0:index]) == item_type
            ).sum()
            item_end = item_length + item_start + 1
            item_source = {
                "p": self._sr._points,
                "mp": self._sr._multipoints,
                "l": self._sr._lines,
                "ml": self._sr._lines,
                "poly": self._sr._polygons,
                "mpoly": self._sr._polygons,
            }[item_type]
            if item_type == "p" or item_type == "mp":
                result = item_source[index]
            else:
                result = item_source[item_start:item_end]
            if item_type == "p":
                return cuPoint(result)
            elif item_type == "mp":
                raise NotImplementedError
            elif item_type == "l":
                return cuLineString(
                    result.to_array().reshape(2 * (item_start - item_end), 2)
                )
            elif item_type == "ml":
                raise NotImplementedError
            elif item_type == "p":
                raise NotImplementedError
            elif item_type == "mp":
                raise NotImplementedError
            else:
                raise TypeError

    @property
    def loc(self):
        """
        Not currently supported.
        """
        return self.GeoSeriesLocIndexer(self)

    @property
    def iloc(self):
        return self.GeoSeriesILocIndexer(self)

    def __getitem__(self, index):
        """
        Returns cuGeometry objects for each of the rows specified by index.
        """
        return self.iloc[index]

    def to_geopandas(self):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        shapely_objs = []
        for geometry in self:
            for point in self._points:
                shapely_objs.append(Point(point.to_array()))
            for line in self._lines:
                shapely_objs.append(LineString(line))
        return gpGeoSeries(shapely_objs)

    def __len__(self):
        """
        Returns the number of unique geometries stored in this cuGeoSeries.
        """
        length = (
            len(self._points.xy) / 2
            + len(self._multipoints.offsets)
            - 1
            + len(self._lines.offsets)
            - 1
            + len(self._polygons.polys)
            - 1
        )
        return int(length)

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


class GpuPoints:
    def __init__(self):
        self.xy = cudf.Series([])
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


class GpuOffset(GpuPoints):
    def __init__(self):
        self.offsets = None

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
            rindex = slice(index, index + 2, 1)
        new_slice = slice(self.offsets[rindex.start], None)
        if rindex.stop < len(self.offsets):
            new_slice = slice(new_slice.start, self.offsets[rindex.stop - 1])
        result = self.xy[new_slice]
        return result


class GpuLines(GpuOffset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        return result


class GpuMultiPoints(GpuOffset):
    pass


class GpuPolygons(GpuOffset):
    def __init__(self):
        self.polys = None
        self.rings = None

    def __repr__(self):
        result = ""
        result += "xy:\n" + self.xy.__repr__() + "\n"
        result += "polys:\n" + self.polys.__repr__() + "\n"
        result += "rings:\n" + self.rings.__repr__() + "\n"
        return result


class cuGeometry:
    def __init__(self, series):
        self.xy = series


class cuPoint(cuGeometry):
    def to_shapely(self):
        return Point(self.xy.reset_index(drop=True))


class cuLineString(cuGeometry):
    def to_shapely(self):
        return LineString(self.xy)
