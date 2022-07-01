# Copyright (c) 2021 NVIDIA CORPORATION

import geopandas as gpd
import pyarrow as pa
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


def getArrowPolygonsType() -> pa.list_:
    return pa.list_(
        pa.field(
            "polygons",
            pa.list_(
                pa.field(
                    "rings",
                    pa.list_(
                        pa.field(
                            "vertices",
                            pa.list_(
                                pa.field("xy", pa.float64(), nullable=False), 2
                            ),
                            nullable=False,
                        )
                    ),
                    nullable=False,
                )
            ),
        )
    )


def getArrowLinestringsType() -> pa.list_:
    return pa.list_(
        pa.field(
            "lines",
            pa.list_(
                pa.field(
                    "offsets",
                    pa.list_(pa.field("xy", pa.float64(), nullable=False), 2),
                    nullable=False,
                )
            ),
            nullable=False,
        )
    )


def getArrowMultiPointsType() -> pa.list_:
    return pa.list_(
        pa.field(
            "points",
            pa.list_(pa.field("xy", pa.float64(), nullable=False), 2),
            nullable=False,
        )
    )


def getArrowPointsType() -> pa.list_:
    return pa.list_(pa.field("xy", pa.float64(), nullable=False), 2)


def getGeoArrowUnionRootType() -> pa.union:
    return pa.union(
        [
            getArrowPointsType(),
            getArrowMultiPointsType(),
            getArrowLinestringsType(),
            getArrowPolygonsType(),
        ],
        mode="dense",
    )


def from_geopandas(geoseries: gpd.GeoSeries) -> pa.lib.UnionArray:
    def get_coordinates(data) -> tuple:
        point_coords = []
        mpoint_coords = []
        line_coords = []
        polygon_coords = []
        all_offsets = [0]
        type_buffer = []

        for geom in data:
            coords = geom.__geo_interface__["coordinates"]
            if isinstance(geom, Point):
                point_coords.append(coords)
                all_offsets.append(1)
            elif isinstance(geom, MultiPoint):
                mpoint_coords.append(coords)
                all_offsets.append(len(coords))
            elif isinstance(geom, LineString):
                line_coords.append([coords])
                all_offsets.append(1)
            elif isinstance(geom, MultiLineString):
                line_coords.append(coords)
                all_offsets.append(len(coords))
            elif isinstance(geom, Polygon):
                polygon_coords.append([coords])
                all_offsets.append(1)
            elif isinstance(geom, MultiPolygon):
                polygon_coords.append(coords)
                all_offsets.append(len(coords))
            else:
                raise TypeError(type(geom))
            type_buffer.append(
                {
                    Point: 0,
                    MultiPoint: 1,
                    LineString: 2,
                    MultiLineString: 3,
                    Polygon: 4,
                    MultiPolygon: 5,
                }[type(geom)]
            )
        return (
            type_buffer,
            point_coords,
            mpoint_coords,
            line_coords,
            polygon_coords,
            all_offsets,
        )

    buffers = get_coordinates(geoseries)
    type_buffer = pa.array(buffers[0]).cast(pa.int8())
    all_offsets = pa.array(buffers[5]).cast(pa.int32())
    children = [
        pa.array(buffers[1], type=getArrowPointsType()),
        pa.array(buffers[2], type=getArrowMultiPointsType()),
        pa.array(buffers[3], type=getArrowLinestringsType()),
        pa.array(buffers[4], type=getArrowPolygonsType()),
    ]

    return pa.UnionArray.from_dense(
        type_buffer,
        all_offsets,
        children,
        ["points", "mpoints", "lines", "polygons"],
    )


class GeoArrow(pa.UnionArray):
    """The GeoArrow specification."""

    def __init__(self, types, offsets, children):
        super().__init__(self, types, offsets, children)

    @property
    def points(self):
        """
        A simple numeric column. x and y coordinates are interleaved such that
        even coordinates are x axis and odd coordinates are y axis.
        """
        return self.fields(0)

    @property
    def multipoints(self):
        """
        Similar to the Points column with the addition of an offsets column.
        The offsets column stores the comparable sizes and coordinates of each
        MultiPoint in the GeoArrowBuffers.
        """
        return self.fields(1)

    @property
    def lines(self):
        """
        Contains the coordinates column, an offsets column,
        and a mlines column.
        The mlines column is optional. The mlines column stores
        the indices of the offsets that indicate the beginning and end of each
        MultiLineString segment. The absence of an `mlines` column indicates
        there are no `MultiLineStrings` in the data source,
        only `LineString` s.
        """
        return self.fields(2)

    @property
    def polygons(self):
        """
        Contains the coordinates column, a rings column specifying
        the beginning and end of every polygon, a polygons column specifying
        the beginning, or exterior, ring of each polygon and the end ring.
        All rings after the first ring are interior rings.  Finally a
        mpolygons column stores the offsets of the polygons that should be
        grouped into MultiPolygons.
        """
        return self.fields(3)

    def copy(self):
        """
        Duplicates the UnionArray
        """
        return pa.UnionArray.from_dense(
            self.type_codes,
            self.offsets,
            [
                self.field(0).values,
                self.field(1).values,
                self.field(2).values,
                self.field(3).values,
            ],
        )

    def to_host(self):
        """
        Return a copy of GeoArrowBuffers backed by host data structures.
        """
        raise NotImplementedError

    def __repr__(self):
        return (
            f"{self.points}\n"
            f"{self.multipoints}\n"
            f"{self.lines}\n"
            f"{self.polygons}\n"
        )
