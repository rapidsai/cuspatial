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
            "multipoints": [0],
            "lines": [0],
            "polygons": {"exterior": [0], "interior": [0]},
        }
        for geometry in geoseries:
            if isinstance(geometry, Point):
                # a single Point geometry will go into the GpuPoints
                # structure. No offsets are required, but an index to the
                # position in the GeoSeries is required.
                current = offsets["points"][-1]
                offsets["points"].append(len(geometry.xy) + current)
            elif isinstance(geometry, MultiPoint):
                # A MultiPoint geometry also is copied into the GpuPoints
                # structure. A MultiPoint object must be created, containing
                # the size of the number of points, the position they are stored
                # in GpuPoints, and the index of the MultiPoint in the
                # GeoSeries.
                #for coord in np.arange(len(geometry)):
                #offsets["multipoints"].append((1 + coord) * 2)
                current = offsets["multipoints"][-1]
                offsets["multipoints"].append(len(geometry) * 2 + current)
            elif isinstance(geometry, LineString):
                # A LineString geometry is stored in the GpuLines structure.
                # Every LineString has a size which is stored in the GpuLines
                # structure. The index of the LineString back into the
                # GeoSeries is also stored.
                current = offsets["lines"][-1]
                offsets["lines"].append(2 * (len(geometry.xy) + current))
            elif isinstance(geometry, MultiLineString):
                # A MultiLineString geometry is stored identically to
                # LineString in the GpuLines structure. The index of the
                # GeoSeries object is also stored.
                current = offsets["lines"][-1]
                mls_lengths = np.array(
                    list(map(lambda x: len(x.coords) * 2, geometry))
                )
                new_offsets = cudf.Series(mls_lengths).cumsum() + current
                offsets["lines"] = offsets["lines"] + list(new_offsets.to_array())
            elif isinstance(geometry, Polygon):
                # A Polygon geometry is stored like a LineString and also
                # contains a buffer of sizes for each inner ring.
                current = offsets["polygons"]["exterior"][-1]
                offsets["polygons"]["exterior"].append(
                    len(geometry.exterior.coords) * 2 + current
                )
                for interior in geometry.interiors:
                    current = offsets["polygons"]["interior"][-1]
                    offsets["polygons"]["interior"].append(
                        len(interior.coords) * 2 + current
                    )
            elif isinstance(geometry, MultiPolygon):
                current = offsets["polygons"]["exterior"][-1]
                mpolys = np.array(
                    list(map(lambda x: len(x.exterior.coords) * 2, geometry))
                )
                new_offsets = cudf.Series(mpolys).cumsum() + current
                offsets["polygons"]["exterior"] = offsets["polygons"]["exterior"] + list(new_offsets.to_array())
        print(offsets)
        """
        offsets["points"] = cudf.Series(np.array(offsets["points"]).flatten())
        offsets["multipoints"] = cudf.Series(
                np.array(offsets["multipoints"]).flatten())
        offsets["lines"] = cudf.Series(np.array(offsets["lines"]).flatten())
        offsets["polygons"]["exterior"] = cudf.Series(np.array(
            offsets["polygons"]["exterior"]
        ).flatten())
        offsets["polygons"]["interior"] = cudf.Series(np.array(
            offsets["polygons"]["interior"]
        ).flatten())
        """
        return offsets

    def _read_geometries(self, geoseries, offsets):
        buffers = {
            "points": cudf.Series(np.zeros(offsets["points"][-1])),
            "multipoints": cudf.Series(np.zeros(offsets["multipoints"][-1])),
            "lines": cudf.Series(np.zeros(offsets["lines"][-1])),
            "polygons": {
                "exterior": cudf.Series(
                    np.zeros(offsets["polygons"]["exterior"][-1])
                ),
                "interior": cudf.Series(
                    np.zeros(offsets["polygons"]["interior"][-1])
                ),
            },
        }
        read_count = {
            "points": 0,
            "multipoints": 0,
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
                multipoint_array = np.array(
                    list(map(lambda x: np.array(x), geometry))
                )
                current_mpoint = 0
                for point in multipoint_array:
                    offset = offsets["multipoints"][current_mpoint]
                    output = buffers["multipoints"]
                    self._cpu_pack_point(np.array(point), offset * 2, output)
                    current_mpoint = current_mpoint + 1
                read_count["multipoints"] = read_count["multipoints"] + 1
            elif isinstance(geometry, LineString):
                self._cpu_pack_linestring(
                    geometry,
                    offsets["lines"][read_count["lines"]],
                    buffers["lines"]
                )
                read_count["lines"] = read_count["lines"] + 1
            elif isinstance(geometry, MultiLineString):
                self._cpu_pack_multilinestring(
                    geometry, offsets["lines"][read_count["lines"]], buffers["lines"]
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
            self._cpu_pack_point(point, offset * 2, output)
            offset = offset + 1

    def _cpu_pack_multilinestring(self, multilinestring, offset, output):
        for linestring in multilinestring:
            self._cpu_pack_linestring(linestring, offset, output)
            offset = offset + 1

    def _cpu_pack_polygon(self, polygon, offset, output):
        for point in polygon.exterior.coords:
            self._cpu_pack_point(point, offset, output)

    def _cpu_pack_multipolygon(self, multipolygon, offset, output):
        for polygon in multipolygon:
            self._cpu_pack_polygon(polygon, offset, output)


class GpuPoints:
    def __init__(self):
        self.xy = cudf.Series([])
        self.z = None
        self.offsets = None
        self.has_z = False
        self._original_series_index = None


class GpuLines(GpuPoints):
    pass


class GeoSeries:
    def __init__(self, geoseries):
        self._reader = GeoSeriesReader(geoseries)
        self._points = GpuPoints()
        self._points.xy = self._reader.buffers[0]["points"]
        self._multipoints = GpuPoints()
        self._multipoints.xy = self._reader.buffers[0]["multipoints"]
        self._multipoints.offsets = self._reader.buffers[1]["multipoints"]
        self._lines = GpuLines()
        self._lines.xy =  self._reader.buffers[0]["lines"]
        self._lines.offsets =  self._reader.buffers[1]["lines"]
        self._polygons = None

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
