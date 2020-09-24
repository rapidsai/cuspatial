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
import pandas as pd

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
                # for coord in np.arange(len(geometry)):
                # offsets["multipoints"].append((1 + coord) * 2)
                current = offsets["multipoints"][-1]
                offsets["multipoints"].append(len(geometry) * 2 + current)
            elif isinstance(geometry, LineString):
                # A LineString geometry is stored in the GpuLines structure.
                # Every LineString has a size which is stored in the GpuLines
                # structure. The index of the LineString back into the
                # GeoSeries is also stored.
                current = offsets["lines"][-1]
                offsets["lines"].append(2 * len(geometry.xy) + current)
            elif isinstance(geometry, MultiLineString):
                # A MultiLineString geometry is stored identically to
                # LineString in the GpuLines structure. The index of the
                # GeoSeries object is also stored.
                current = offsets["lines"][-1]
                mls_lengths = np.array(
                    list(map(lambda x: len(x.coords) * 2, geometry))
                )
                new_offsets = cudf.Series(mls_lengths).cumsum() + current
                offsets["lines"] = offsets["lines"] + list(
                    new_offsets.to_array()
                )
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
                        offsets["polygons"]["rings"].append(
                            len(interior.coords) * 2 + rings_current
                        )
                        num_rings = num_rings + 1
                    offsets["polygons"]["polygons"].append(num_rings + current)
        return offsets

    def _read_geometries(self, geoseries, offsets):
        buffers = {
            "points": cudf.Series(np.zeros(offsets["points"][-1])),
            "multipoints": cudf.Series(np.zeros(offsets["multipoints"][-1])),
            "lines": cudf.Series(np.zeros(offsets["lines"][-1])),
            "polygons": {
                "polygons": cudf.Series(
                    np.zeros(len(offsets["polygons"]["polygons"]))
                ),
                "rings": cudf.Series(
                    np.zeros(len(offsets["polygons"]["rings"]))
                ),
                "coords": cudf.Series(
                    np.zeros(offsets["polygons"]["rings"][-1])
                ),
            },
        }
        read_count = {
            "points": 0,
            "multipoints": 0,
            "lines": 0,
            "polygons": 0,
        }
        for geometry in geoseries:
            if isinstance(geometry, Point):
                # write a point to the points buffer
                # increase read_count of points pass
                p = np.array(geometry)
                i = read_count["points"] * 2
                buffers["points"][i] = p[0]
                buffers["points"][i + 1] = p[1]
                read_count["points"] = read_count["points"] + 1
            elif isinstance(geometry, MultiPoint):
                for point in geometry:
                    p = np.array(point)
                    i = read_count["multipoints"] * 2
                    buffers["multipoints"][i] = p[0]
                    buffers["multipoints"][i + 1] = p[1]
                    read_count["multipoints"] = read_count["multipoints"] + 1
            elif isinstance(geometry, LineString):
                line = np.array(geometry.xy).T
                for point in line:
                    p = np.array(point)
                    i = read_count["lines"] * 2
                    buffers["lines"][i] = p[0]
                    buffers["lines"][i + 1] = p[1]
                    read_count["lines"] = read_count["lines"] + 1
            elif isinstance(geometry, MultiLineString):
                for linestring in geometry:
                    line = np.array(linestring.xy).T
                    for point in line:
                        p = np.array(point)
                        i = read_count["lines"] * 2
                        buffers["lines"][i] = p[0]
                        buffers["lines"][i + 1] = p[1]
                        read_count["lines"] = read_count["lines"] + 1
            elif isinstance(geometry, Polygon):
                # copy exterior
                exterior = geometry.exterior.coords
                interiors = geometry.interiors
                for point in exterior:
                    p = np.array(point)
                    i = read_count["polygons"] * 2
                    buffers["polygons"]["coords"][i] = p[0]
                    buffers["polygons"]["coords"][i + 1] = p[1]
                    read_count["polygons"] = read_count["polygons"] + 1
                for interior in interiors:
                    for ipoint in interior.coords:
                        ip = np.array(ipoint)
                        i = read_count["polygons"] * 2
                        buffers["polygons"]["coords"][i] = ip[0]
                        buffers["polygons"]["coords"][i + 1] = ip[1]
                        read_count["polygons"] = read_count["polygons"] + 1
            elif isinstance(geometry, MultiPolygon):
                for polygon in geometry:
                    exterior = polygon.exterior.coords
                    interiors = polygon.interiors
                    for point in exterior:
                        p = np.array(point)
                        i = read_count["polygons"] * 2
                        buffers["polygons"]["coords"][i] = p[0]
                        buffers["polygons"]["coords"][i + 1] = p[1]
                        read_count["polygons"] = read_count["polygons"] + 1
                    for interior in interiors:
                        for ipoint in interior.coords:
                            ip = np.array(ipoint)
                            i = read_count["polygons"] * 2
                            buffers["polygons"]["coords"][i] = ip[0]
                            buffers["polygons"]["coords"][i + 1] = ip[1]
                            read_count["polygons"] = read_count["polygons"] + 1
            else:
                raise NotImplementedError
        offsets["polygons"]["rings"] = cudf.Series(
            offsets["polygons"]["rings"]
        )
        offsets["polygons"]["polygons"] = cudf.Series(
            offsets["polygons"]["polygons"]
        )
        offsets["lines"] = cudf.Series(offsets["lines"])
        offsets["points"] = cudf.Series(offsets["points"])
        offsets["multipoints"] = cudf.Series(offsets["multipoints"])
        return (buffers, offsets)


class GpuPoints:
    def __init__(self):
        self.xy = cudf.Series([])
        self.z = None
        self.offsets = None
        self.has_z = False
        self._original_series_index = None


class GpuLines(GpuPoints):
    pass


class GpuPolygons(GpuPoints):
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
    def __init__(self, geoseries):
        self._reader = GeoSeriesReader(geoseries)
        self._points = GpuPoints()
        self._points.xy = self._reader.buffers[0]["points"]
        self._multipoints = GpuPoints()
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

    def to_geopandas(self):
        shapely_objs = np.zeros(len(self))
        current = 0
        for point in self._points:
            shapely_objs[current] = Point(self._points[current])
            current = current + 1
        for multipoint in self._multipoints:
            shapely_objs[current] = MultiPoint(self._multipoint[current])
            current = current + 1



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
