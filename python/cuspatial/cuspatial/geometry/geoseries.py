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
            "polygons": {"polygons": [0], "rings": [0]},
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
                num_rings = 1
                rings_current = offsets["polygons"]["rings"][-1]
                offsets["polygons"]["rings"].append(
                    len(geometry.exterior.coords) * 2 + rings_current
                )
                for interior in geometry.interiors:
                    rings_current = offsets["polygons"]["rings"][-1]
                    offsets["polygons"]["rings"].append(
                        len(interior.coords) * 2 + rings_current
                    )
                    num_rings = num_rings + 1
                current = offsets["polygons"]["polygons"][-1]
                offsets["polygons"]["polygons"].append(num_rings + current)
            elif isinstance(geometry, MultiPolygon):
                for poly in geometry:
                    current = offsets["polygons"]["polygons"][-1]
                    num_rings = 1
                    rings_current = offsets["polygons"]["rings"][-1]
                    offsets["polygons"]["rings"].append(
                        len(poly.exterior.coords) * 2 + rings_current
                    )
                    for interior in poly.interiors:
                        rings_current = offsets["polygons"]["rings"][-1]
                        print(rings_current)
                        print(offsets["polygons"]["rings"])
                        offsets["polygons"]["rings"].append(
                            len(interior.coords) *  2 + rings_current
                        )
                        num_rings = num_rings + 1
                    offsets["polygons"]["polygons"].append(num_rings + current)
                """
                mpolys = np.array(
                    list(map(lambda x: len(x.exterior.coords) * 2, geometry))
                )
                new_offsets = cudf.Series(mpolys).cumsum() + current
                offsets["polygons"]["polygons"] = offsets["polygons"]["polygons"] + list(new_offsets.to_array())
                for polygon in geometry:
                    for interior in polygon.interiors:
                        current = offsets["polygons"]["polygons"][-1]
                        offsets["polygons"]["rings"].append(
                            len(interior.coords)  * 2 + current
                        )
                """
        print(offsets)
        """
        offsets["points"] = cudf.Series(np.array(offsets["points"]).flatten())
        offsets["multipoints"] = cudf.Series(
                np.array(offsets["multipoints"]).flatten())
        offsets["lines"] = cudf.Series(np.array(offsets["lines"]).flatten())
        offsets["polygons"]["polygons"] = cudf.Series(np.array(
            offsets["polygons"]["polygons"]
        ).flatten())
        offsets["polygons"]["rings"] = cudf.Series(np.array(
            offsets["polygons"]["rings"]
        ).flatten())
        """
        return offsets

    def _read_geometries(self, geoseries, offsets):
        buffers = {
            "points": cudf.Series(np.zeros(offsets["points"][-1])),
            "multipoints": cudf.Series(np.zeros(offsets["multipoints"][-1])),
            "lines": cudf.Series(np.zeros(offsets["lines"][-1])),
            "polygons": {
                "polygons": cudf.Series(
                    np.zeros(offsets["polygons"]["polygons"][-1])
                ),
                "rings": cudf.Series(
                    np.zeros(offsets["polygons"]["rings"][-1])
                ),
            },
        }
        read_count = {
            "points": 0,
            "multipoints": 0,
            "lines": 0,
            "polygons": {"polygons": 0, "rings": 0},
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
                    read_count["polygons"]["polygons"],
                    buffers["polygons"]["polygons"],
                )
                read_count["polygons"]["polygons"] = (
                    read_count["polygons"]["polygons"] + 1
                )
            elif isinstance(geometry, MultiPolygon):
                self._cpu_pack_multipolygon(
                    geometry,
                    read_count["polygons"]["polygons"],
                    buffers["polygons"]["polygons"],
                )
                read_count["polygons"]["polygons"] = (
                    read_count["polygons"]["polygons"] + 1
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
