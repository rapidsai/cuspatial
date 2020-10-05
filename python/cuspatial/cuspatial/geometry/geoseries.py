# 2020 NVIDIA

from geopandas.geoseries import GeoSeries as gpGeoSeries
import pandas as pd

from shapely.geometry import (
    Point,
    LineString,
)

import cudf


class GpuPoints:
    def __init__(self):
        self.xy = cudf.Series([])
        self.z = None
        self.has_z = False
        self._original_series_index = None

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
            raise IterableError
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


class GeoSeries:
    def __init__(self, reader):
        self._reader = reader
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
            self._reader.offsets["polygons"]["polygons"]
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
            # Points are the only structure that stores point and multipoint in separate
            # arrays. I'm not suire why we decided to do this anymore.
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
        return self.GeoSeriesLocIndexer(self)

    @property
    def iloc(self):
        return self.GeoSeriesILocIndexer(self)

    def __getitem__(self, index):
        return self.iloc[index]

    def to_geopandas(self):
        shapely_objs = []
        for geometry in self:
            for point in self._points:
                shapely_objs.append(Point(point.to_array()))
            for multipoint in self._multipoints:
                shapely_objs.append(MultiPoint(multipoint))
            for line in self._lines:
                shapely_objs.append(LineString(line))
        return gpGeoSeries(shapely_objs)

    def __len__(self):
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


class cuGeometry:
    def __init__(self, series):
        self.xy = series


class cuPoint(cuGeometry):
    def to_shapely(self):
        return Point(self.xy.reset_index(drop=True))


class cuLineString(cuGeometry):
    def to_shapely(self):
        return LineString(self.xy)
