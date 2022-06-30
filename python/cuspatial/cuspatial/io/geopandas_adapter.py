# Copyright (c) 2020-2021 NVIDIA CORPORATION.

import numpy as np
import pyarrow as pa
from geopandas import GeoSeries as gpGeoSeries
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import pygeoarrow


class GeoPandasAdapter:
    def __init__(self, geoseries: gpGeoSeries):
        """
        GeoPandasAdapter copies a GeoPandas GeoSeries object iteratively into
        a set of arrays: points, multipoints, lines, and polygons.

        Parameters
        ----------
        geoseries : A GeoPandas GeoSeries
        """

        self.buffers = self._read_geopandas_to_geoarrow(geoseries)

    def _read_geopandas_to_geoarrow(geoseries: gpGeoSeries) -> pygeoarrow.union_
        point_coords = []
        mpoint_coords = []
        mpoint_offsets = [0]
        line_coords = []
        line_offsets = [0]
        polygon_coords = []
        polygon_offsets = [0]
        all_coords = []
        all_offsets = [0]
        type_buffer = []
    

        for geom in data:
            coords = geom.__geo_interface__["coordinates"]
            if isinstance(geom, Point):
                point_coords.append(coords)
            elif isinstance(geom, MultiPoint):
                mpoint_coords.append(coords)
                mpoint_offsets.append(mpoint_offsets[-1] + len(mpoint_coords))
            elif isinstance(geom, LineString):
                line_coords.append([coords])
                line_offsets.append(line_offsets[-1] + len(line_coords))
            elif isinstance(geom, MultiLineString):
                line_coords.append(coords)
                line_offsets.append(line_offsets[-1] + len(line_coords))
            elif isinstance(geom, Polygon):
                polygon_coords.append([coords])
                polygon_offsets.append(polygon_offsets[-1] + len(polygon_coords))
            elif isinstance(geom, MultiPolygon) or isinstance(geom, Polygon):
                polygon_coords.append(coords)
                polygon_offsets.append(polygon_offsets[-1] + len(polygon_coords))
            else:
                raise TypeError(type(geom))
            all_coords.append(coords)
            all_offsets.append(all_offsets[-1] + len(all_coords[-1]))
            type_buffer.append({
                Point: 0,
                MultiPoint: 1,
                LineString: 2,
                MultiLineString: 2,
                Polygon: 3,
                MultiPolygon: 3
            }[type(geom)])

        return pygeoarrow.DenseUnion(
            type_buffer,
            all_offsets,
                children,
                ["points", "mpoints", "lines", "polygons"]
        )

    def _load_geometry_offsets(self, geoseries: gpGeoSeries) -> dict:
        """
        Computes the offet arrays and buffer sizes  that will be required
        to store the geometries.

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

    def _read_geometries(
        self,
        geoseries: gpGeoSeries,
        offsets: dict,
    ) -> dict:
        """
        Creates a set of buffers sized to fit all of the geometries and
        iteratively populates them with geometry coordinate values.

        Parameters
        ----------
        geoseries : A GeoPandas GeoSeries object.
        offsets : The set of offsets that correspond to the geoseries argument.
        """
        buffers = {
            "points": np.zeros(offsets["points"][-1]),
            "multipoints": np.zeros(offsets["multipoints"][-1]),
            "lines": np.zeros(offsets["lines"][-1]),
            "polygons": {
                "polygons": np.zeros(len(offsets["polygons"]["polygons"])),
                "rings": np.zeros(len(offsets["polygons"]["rings"])),
                "coords": np.zeros(offsets["polygons"]["rings"][-1]),
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
                i = read_count["points"] * 2
                buffers["points"][i] = geometry.x
                buffers["points"][i + 1] = geometry.y
                read_count["points"] = read_count["points"] + 1
                input_types.append("p")
                input_lengths.append(1)
                inputs.append({"type": "p", "length": 1})
            elif isinstance(geometry, MultiPoint):
                points = np.array(geometry)
                size = points.shape[0] * 2
                i = read_count["multipoints"]
                buffers["multipoints"][slice(i, i + size, 2)] = points[:, 0]
                buffers["multipoints"][slice(i + 1, i + size, 2)] = points[
                    :, 1
                ]
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
        return {
            "buffers": buffers,
            "input_types": input_types,
            "input_lengths": input_lengths,
            "inputs": inputs,
        }

    def get_geoarrow_host_buffers(self) -> dict:
        """
        Returns a set of host buffers containing the geopandas object converted
        to GeoArrow format.
        """
        points_xy = []
        mpoints_xy = []
        mpoints_offsets = []
        lines_xy = []
        lines_offsets = []
        mlines = []
        polygons_xy = []
        polygons_polygons = []
        polygons_rings = []
        mpolygons = []
        buffers = self.buffers["buffers"]
        points_xy = buffers["points"]
        mpoints_xy = buffers["multipoints"]
        mpoints_offsets = self.offsets["multipoints"]
        lines_xy = buffers["lines"]
        lines_offsets = self.offsets["lines"]
        mlines = self.offsets["mlines"]
        polygons_xy = buffers["polygons"]["coords"]
        polygons_polygons = self.offsets["polygons"]["polygons"]
        polygons_rings = self.offsets["polygons"]["rings"]
        mpolygons = self.offsets["polygons"]["mpolys"]
        return {
            "points_xy": points_xy,
            "mpoints_xy": mpoints_xy,
            "mpoints_offsets": mpoints_offsets,
            "lines_xy": lines_xy,
            "lines_offsets": lines_offsets,
            "mlines": mlines,
            "polygons_xy": polygons_xy,
            "polygons_polygons": polygons_polygons,
            "polygons_rings": polygons_rings,
            "mpolygons": mpolygons,
        }

    def get_geopandas_meta(self) -> dict:
        """
        Returns the metadata that was created converting the GeoSeries into
        GeoArrow format. The metadata essentially contains the object order
        in the GeoSeries format. GeoArrow doesn't support custom orderings,
        every GeoArrow data store contains points, multipoints, lines, and
        polygons in an arbitrary order.
        """
        buffers = self.buffers
        return {
            "input_types": buffers["input_types"],
            "input_lengths": buffers["input_lengths"],
            "inputs": buffers["inputs"],
        }
