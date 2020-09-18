# 2020 NVIDIA
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)

import numpy as np

import cudf


class GeoSeriesReader:
    def __init__(self, geoseries):
        self.offsets = self._load_geometry_offsets(geoseries)
        self.buffers = self._read_geometries(geoseries, self.offsets)

    def _load_geometry_offsets(self, geoseries):
        offsets = {
            "points": [0],
            "lines": [0],
            "polygons": {"exterior": [0], "interior": [0]},
        }
        current = 0
        for geometry in geoseries:
            if isinstance(geometry, Point):
                offsets["points"].append(len(geometry.xy) + current)
                current = offsets["points"][len(offsets["points"]) - 1]
            elif isinstance(geometry, MultiPoint):
                for coord in np.arange(len(geometry)):
                    offsets["points"].append((1 + coord) * 2)
                current = offsets["points"][len(offsets["points"]) - 1]
            elif isinstance(geometry, LineString):
                offsets["lines"].append((1 + np.arange(len(geometry.xy))) * 2)
            elif isinstance(geometry, MultiLineString):
                mls_lengths = np.array(
                    list(map(lambda x: len(x.coords) * 2, geometry))
                )
                offsets["lines"].append(
                    cudf.Series(mls_lengths).cumsum().to_array()
                )
            elif isinstance(geometry, Polygon):
                offsets["polygons"]["exterior"].append(
                    len(geometry.exterior.coords) * 2
                )
            elif isinstance(geometry, MultiPolygon):
                mpolys = np.array(
                    list(map(lambda x: len(x.exterior.coords) * 2, geometry))
                )
                offsets["polygons"]["exterior"].append(
                    cudf.Series(mpolys).cumsum().to_array()
                )
        offsets["points"] = np.array(offsets["points"]).flatten()
        offsets["lines"] = np.array(offsets["lines"]).flatten()
        offsets["polygons"]["exterior"] = np.array(
            offsets["polygons"]["exterior"]
        ).flatten()
        offsets["polygons"]["interior"] = np.array(
            offsets["polygons"]["interior"]
        ).flatten()
        return offsets

    def _read_geometries(self, geoseries, offsets):
        buffers = {
            "points": cudf.Series(np.zeros(offsets["points"].max())),
            "lines": cudf.Series(np.zeros(offsets["lines"].max())),
            "polygons": {
                "exterior": cudf.Series(
                    np.zeros(offsets["polygons"]["exterior"].max())
                ),
                "interior": cudf.Series(
                    np.zeros(offsets["polygons"]["interior"].max())
                ),
            },
        }
        read_count = {
            "points": 0,
            "lines": 0,
            "polygons": {"exterior": 0, "interior": 0},
        }
        for geometry in geoseries:
            if isinstance(geometry, Point):
                self._cpu_pack_point(
                    geometry.xy,
                    offsets["points"][read_count["points"]],
                    buffers["points"],
                )
                read_count["points"] = read_count["points"] + 1
            elif isinstance(geometry, MultiPoint):
                breakpoint()
                self._cpu_pack_multipoint(
                    geometry, offsets["points"][read_count["points"]],
                    buffers["points"]
                )
                read_count["points"] = read_count["points"] + 1
            elif isinstance(geometry, LineString):
                self._cpu_pack_linestring(
                    geometry, read_count["lines"], buffers["lines"]
                )
                read_count["lines"] = read_count["lines"] + 1
            elif isinstance(geometry, MultiLineString):
                self._cpu_pack_multilinestring(
                    geometry, read_count["lines"], buffers["lines"]
                )
                read_count["lines"] = read_count["lines"] + 1
            elif isinstance(geometry, Polygon):
                self._cpu_pack_polygon(
                    geometry,
                    read_count["polygons"]["exterior"],
                    buffers["polygons"]["exterior"],
                )
                read_count["polygons"]["exterior"] = (
                    read_count["polygons"]["exterior"] + 1
                )
            elif isinstance(geometry, MultiPolygon):
                self._cpu_pack_multipolygon(
                    geometry,
                    read_count["polygons"]["exterior"],
                    buffers["polygons"]["exterior"],
                )
                read_count["polygons"]["exterior"] = (
                    read_count["polygons"]["exterior"] + 1
                )
            else:
                raise NotImplementedError
        return (buffers, offsets)

    def _cpu_pack_point(self, point, offset, output):
        output[0 + offset] = point[0]
        output[1 + offset] = point[1]

    def _cpu_pack_multipoint(self, multipoint, offset, output):
        multipoint_array = np.array(
            list(map(lambda x: np.array(x), multipoint))
        )
        for point in multipoint_array:
            self._cpu_pack_point(np.array(point), offset, output)

    def _cpu_pack_linestring(self, linestring, offset, output):
        linestring_array = np.array(
            list(map(lambda x: np.array(x), linestring.coords))
        )
        for point in linestring_array:
            self._cpu_pack_point(point, offset, output)

    def _cpu_pack_multilinestring(self, multilinestring, offset, output):
        for linestring in multilinestring:
            self._cpu_pack_linestring(linestring, offset, output)

    def _cpu_pack_polygon(self, polygon, offset, output):
        for point in polygon.exterior.coords:
            self._cpu_pack_point(point, offset, output)

    def _cpu_pack_multipolygon(self, multipolygon, offset, output):
        for polygon in multipolygon:
            self._cpu_pack_polygon(polygon, offset, output)


class GpuPoints:
    def __init__(self):
        self.coords = cudf.Series([])
        self.offsets = cudf.Series([])


class GpuLines(GpuPoints):
    def __init__(self):
        super(GpuPoints, self).__init__()


class GeoSeries:
    def __init__(self, geoseries):
        self._reader = GeoSeriesReader(geoseries)
        self._points = GpuPoints()
        self._points.coords = self._reader.buffers[0]["points"]
        self._points.offsets = self._reader.buffers[1]["points"]
        self._lines = GpuLines()
        self._polygons = None

    @property
    def points(self):
        return self._points

    @property
    def lines(self):
        return self._lines

    @property
    def polygons(self):
        return self._polygons
