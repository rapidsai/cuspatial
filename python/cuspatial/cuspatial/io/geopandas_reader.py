# Copyright (c) 2020-2022 NVIDIA CORPORATION.

from geopandas import GeoSeries as gpGeoSeries
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
)

import cudf

from cuspatial.core._column.geometa import Feature_Enum
from cuspatial.io import pygeoarrow

NONE_OFFSET = -1


def parse_geometries(geoseries: gpGeoSeries) -> tuple:
    point_coords = []
    mpoint_coords = []
    line_coords = []
    polygon_coords = []
    all_offsets = []
    type_buffer = []
    point_offsets = [0]
    mpoint_offsets = [0]
    line_offsets = [0]
    polygon_offsets = [0]

    for geom in geoseries:
        if geom is None:
            all_offsets.append(NONE_OFFSET)
            type_buffer.append(Feature_Enum.NONE.value)
            continue
        coords = mapping(geom)["coordinates"]
        if isinstance(geom, Point):
            point_coords.append(coords)
            all_offsets.append(point_offsets[-1])
            point_offsets.append(point_offsets[-1] + 1)
            type_buffer.append(Feature_Enum.POINT.value)
        elif isinstance(geom, MultiPoint):
            mpoint_coords.append(coords)
            all_offsets.append(mpoint_offsets[-1])
            mpoint_offsets.append(mpoint_offsets[-1] + 1)
            type_buffer.append(Feature_Enum.MULTIPOINT.value)
        elif isinstance(geom, LineString):
            line_coords.append([coords])
            all_offsets.append(line_offsets[-1])
            line_offsets.append(line_offsets[-1] + 1)
            type_buffer.append(Feature_Enum.LINESTRING.value)
        elif isinstance(geom, MultiLineString):
            line_coords.append(coords)
            all_offsets.append(line_offsets[-1])
            line_offsets.append(line_offsets[-1] + 1)
            type_buffer.append(Feature_Enum.LINESTRING.value)
        elif isinstance(geom, Polygon):
            polygon_coords.append([coords])
            all_offsets.append(polygon_offsets[-1])
            polygon_offsets.append(polygon_offsets[-1] + 1)
            type_buffer.append(Feature_Enum.POLYGON.value)
        elif isinstance(geom, MultiPolygon):
            polygon_coords.append(coords)
            all_offsets.append(polygon_offsets[-1])
            polygon_offsets.append(polygon_offsets[-1] + 1)
            type_buffer.append(Feature_Enum.POLYGON.value)
        else:
            raise TypeError(type(geom))
    return (
        type_buffer,
        all_offsets,
        point_coords,
        mpoint_coords,
        line_coords,
        polygon_coords,
    )


class GeoPandasReader:
    buffers = None
    source = None

    def __init__(self, geoseries: gpGeoSeries):
        """
        GeoPandasReader copies a GeoPandas GeoSeries object iteratively into
        a set of arrays: points, multipoints, lines, and polygons.

        Parameters
        ----------
        geoseries : A GeoPandas GeoSeries
        """
        self.buffers = pygeoarrow.from_lists(*parse_geometries(geoseries))

    def _get_geotuple(self) -> cudf.Series:
        """
        TODO:
        Returns the four basic cudf.ListSeries objects for
        points, mpoints, lines, and polygons
        """
        points = cudf.Series.from_arrow(
            self.buffers.field(Feature_Enum.POINT.value)
        )
        mpoints = cudf.Series.from_arrow(
            self.buffers.field(Feature_Enum.MULTIPOINT.value)
        )
        lines = cudf.Series.from_arrow(
            self.buffers.field(Feature_Enum.LINESTRING.value)
        )
        polygons = cudf.Series.from_arrow(
            self.buffers.field(Feature_Enum.POLYGON.value)
        )
        return (
            points,
            mpoints,
            lines,
            polygons,
        )

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
            "input_types": buffers.type_codes,
            "union_offsets": buffers.offsets,
        }
