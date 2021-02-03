# Copyright (c) 2020 NVIDIA CORPORATION.

from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)

import cupy as cp

import cudf


class GeoSeriesReader:
    def __init__(self, geoseries, interleaved=True):
        """
        GeoSeriesReader copies a GeoPandas GeoSeries object iteratively into
        a set of GeoSeries buffers: points, multipoints, lines, and polygons.

        Parameters
        ----------
        geoseries : A GeoPandas GeoSeries
        interleaved: Boolean
            Return buffers from _read_geometries that have interleaved x,y
            coordinates in a single GPU buffer, or use separate buffers for
            x and y.
        """
        self.interleaved = interleaved
        self.offsets = self._load_geometry_offsets(geoseries)
        self.buffers = self._read_geometries(geoseries, self.offsets)

    def _load_geometry_offsets(self, geoseries):
        """
        Precomputes the buffers that will be required to store the geometries.

        Parameters
        ----------
        geoseries : A GeoPandas GeoSeries
        """
        offsets = {
            "points": [0],
            "multipoints": [0],
            "lines": [0],
            "mlines": [],
            "polygons": {"polygons": [0], "rings": [0], "mpolys": []},
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
                # the size of the number of points, the position they are
                # stored in GpuPoints, and the index of the MultiPoint in the
                # GeoSeries.
                # for coord in cp.arange(len(geometry)):
                # offsets["multipoints"].append((1 + coord) * 2)
                current = offsets["multipoints"][-1]
                offsets["multipoints"].append(len(geometry) * 2 + current)
            elif isinstance(geometry, LineString):
                # A LineString geometry is stored in the GpuLines structure.
                # Every LineString has a size which is stored in the GpuLines
                # structure. The index of the LineString back into the
                # GeoSeries is also stored.
                current = offsets["lines"][-1]
                offsets["lines"].append(2 * len(geometry.coords) + current)
            elif isinstance(geometry, MultiLineString):
                # A MultiLineString geometry is stored identically to
                # LineString in the GpuLines structure. The index of the
                # GeoSeries object is also stored.
                offsets["mlines"].append(len(offsets["lines"]) - 1)
                for linestring in geometry:
                    current = offsets["lines"][-1]
                    offsets["lines"].append(
                        2 * len(linestring.coords) + current
                    )
                offsets["mlines"].append(len(offsets["lines"]) - 1)
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
                current = offsets["polygons"]["polygons"][-1]
                offsets["polygons"]["mpolys"].append(
                    len(offsets["polygons"]["polygons"]) - 1
                )
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
                offsets["polygons"]["mpolys"].append(
                    len(offsets["polygons"]["polygons"]) - 1
                )
        return offsets

    def _read_geometries(self, geoseries, offsets):
        """
        Creates a set of buffers sized to fit all of the geometries and
        iteratively populates them with geometry coordinate values.

        Parameters
        ----------
        geoseries : A GeoPandas GeoSeries object.
        offsets : The set of offsets that correspond to the geoseries argument.
        """
        buffers = {
            "points": cudf.Series(cp.zeros(offsets["points"][-1])),
            "multipoints": cudf.Series(cp.zeros(offsets["multipoints"][-1])),
            "lines": cudf.Series(cp.zeros(offsets["lines"][-1])),
            "polygons": {
                "polygons": cudf.Series(
                    cp.zeros(len(offsets["polygons"]["polygons"]))
                ),
                "rings": cudf.Series(
                    cp.zeros(len(offsets["polygons"]["rings"]))
                ),
                "coords": cudf.Series(
                    cp.zeros(offsets["polygons"]["rings"][-1])
                ),
            },
        }
        read_count = {
            "points": 0,
            "multipoints": 0,
            "lines": 0,
            "polygons": 0,
        }
        inputs = []
        input_types = []
        input_lengths = []
        for geometry in geoseries:
            if isinstance(geometry, Point):
                # write a point to the points buffer
                # increase read_count of points pass
                p = cp.array(geometry)
                i = read_count["points"] * 2
                buffers["points"][i] = p[0]
                buffers["points"][i + 1] = p[1]
                read_count["points"] = read_count["points"] + 1
                input_types.append("p")
                input_lengths.append(1)
                inputs.append({"type": "p", "length": 1})
            elif isinstance(geometry, MultiPoint):
                points = cp.array(geometry).T
                size = len(points.T) * 2
                i = read_count["multipoints"]
                buffers["multipoints"][slice(i, i + size, 2)] = points[0]
                buffers["multipoints"][slice(i + 1, i + size, 2)] = points[1]
                read_count["multipoints"] = read_count["multipoints"] + size
                input_types.append("mp")
                input_lengths.append(len(geometry))
                inputs.append({"type": "mp", "length": len(geometry)})
            elif isinstance(geometry, LineString):
                size = len(geometry.xy[0]) * 2
                i = read_count["lines"]
                buffers["lines"][slice(i, i + size, 2)] = geometry.xy[0]
                buffers["lines"][slice(i + 1, i + size, 2)] = geometry.xy[1]
                read_count["lines"] = read_count["lines"] + size
                input_types.append("l")
                input_lengths.append(1)
                inputs.append({"type": "l", "length": 1})
            elif isinstance(geometry, MultiLineString):
                substrings = []
                for linestring in geometry:
                    size = len(linestring.xy[0]) * 2
                    i = read_count["lines"]
                    buffers["lines"][slice(i, i + size, 2)] = linestring.xy[0]
                    buffers["lines"][
                        slice(i + 1, i + size, 2)
                    ] = linestring.xy[1]
                    read_count["lines"] = read_count["lines"] + size
                    substrings.append({"type": "l", "length": size})
                input_types.append("ml")
                input_lengths.append(len(geometry))
                inputs.append(
                    {
                        "type": "ml",
                        "length": len(geometry),
                        "children": substrings,
                    }
                )
            elif isinstance(geometry, Polygon):
                # copy exterior
                exterior = geometry.exterior.coords.xy
                size = len(exterior[0]) * 2
                i = read_count["polygons"]
                buffers["polygons"]["coords"][
                    slice(i, i + size, 2)
                ] = exterior[0]
                buffers["polygons"]["coords"][
                    slice(i + 1, i + size, 2)
                ] = exterior[1]
                read_count["polygons"] = read_count["polygons"] + size
                interiors = geometry.interiors
                for interior in interiors:
                    interior_coords = interior.coords.xy
                    size = len(interior_coords[0]) * 2
                    i = read_count["polygons"]
                    buffers["polygons"]["coords"][
                        slice(i, i + size, 2)
                    ] = interior_coords[0]
                    buffers["polygons"]["coords"][
                        slice(i + 1, i + size, 2)
                    ] = interior_coords[1]
                    read_count["polygons"] = read_count["polygons"] + size
                input_types.append("poly")
                input_lengths.append(1)
                inputs.append({"type": "poly", "length": 1})
            elif isinstance(geometry, MultiPolygon):
                subpolys = []
                for polygon in geometry:
                    exterior = polygon.exterior.coords.xy
                    size = len(exterior[0]) * 2
                    i = read_count["polygons"]
                    buffers["polygons"]["coords"][
                        slice(i, i + size, 2)
                    ] = exterior[0]
                    buffers["polygons"]["coords"][
                        slice(i + 1, i + size, 2)
                    ] = exterior[1]
                    read_count["polygons"] = read_count["polygons"] + size
                    interiors = polygon.interiors
                    for interior in interiors:
                        interior_coords = interior.coords.xy
                        size = len(interior_coords[0]) * 2
                        i = read_count["polygons"]
                        buffers["polygons"]["coords"][
                            slice(i, i + size, 2)
                        ] = interior_coords[0]
                        buffers["polygons"]["coords"][
                            slice(i + 1, i + size, 2)
                        ] = interior_coords[1]
                        read_count["polygons"] = read_count["polygons"] + size
                    subpolys.append({"type": "poly", "length": 1})
                input_types.append("mpoly")
                input_lengths.append(len(geometry))
                inputs.append(
                    {
                        "type": "ml",
                        "length": len(geometry),
                        "children": subpolys,
                    }
                )
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
        return (buffers, offsets, input_types, input_lengths, inputs)
