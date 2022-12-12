import cupy as cp

from cudf.core.column import as_column

import cuspatial._lib.nearest_points as nearest_points
from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.geodataframe import GeoDataFrame
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_points,
)


def pairwise_point_linestring_nearest_points(
    points: GeoSeries, linestrings: GeoSeries
) -> GeoDataFrame:
    """Returns the nearest points between two GeoSeries of points and
    linestrings.

    Multipoints and Multilinestrings are also supported. With restriction that
    the `points` series must contain either only points or only multipoints.

    Parameters
    ----------
    points : GeoSeries
        A GeoSeries of points. Currently either only a series of points or
        a series of multipoints is supported.
    linestrings : GeoSeries
        A GeoSeries of linestrings or multilinestrings.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with four columns.

        - "point_geometry_id" contains index of the nearest point in the row.
          If `points` consists of single-points, it is always 0.
        - "linestring_geometry_id" contains the index of the linestring in the
          multilinestring that contains the nearest point.
        - "segment_id" contains the index of the segment in the linestring that
          contains the nearest point.
        - "geometry" contains the points of the nearest
          point on the linestring.
    """

    if len(points) != len(linestrings):
        raise ValueError(
            "The inputs should have the same number of geometries"
        )

    if len(points) == 0:
        data = {
            "point_geometry_id": [],
            "linestring_geometry_id": [],
            "segment_id": [],
            "geometry": GeoSeries([]),
        }
        return GeoDataFrame._from_data(data)

    if not contains_only_points(points):
        raise ValueError("`points` must contain only point geometries.")

    if not contains_only_linestrings(linestrings):
        raise ValueError(
            "`linestrings` must contain only linestring geometries."
        )

    if len(points.points.xy) > 0 and len(points.multipoints.xy) > 0:
        raise NotImplementedError(
            "Mixing points and multipoint geometries in `points` is not yet "
            "supported."
        )

    points_xy = (
        points.points.xy
        if len(points.points.xy) > 0
        else points.multipoints.xy
    )
    points_geometry_offset = (
        None
        if len(points.points.xy) > 0
        else as_column(points.multipoints.geometry_offset)
    )

    (
        point_geometry_id,
        linestring_geometry_id,
        segment_id,
        point_on_linestring_xy,
    ) = nearest_points.pairwise_point_linestring_nearest_points(
        points_xy._column,
        as_column(linestrings.lines.part_offset),
        linestrings.lines.xy._column,
        points_geometry_offset,
        as_column(linestrings.lines.geometry_offset),
    )

    point_on_linestring = GeoColumn._from_points_xy(point_on_linestring_xy)
    nearest_points_on_linestring = GeoSeries(point_on_linestring)

    if not point_geometry_id:
        point_geometry_id = cp.zeros(len(points), dtype=cp.int32)

    data = {
        "point_geometry_id": point_geometry_id,
        "linestring_geometry_id": linestring_geometry_id,
        "segment_id": segment_id,
        "geometry": nearest_points_on_linestring,
    }

    return GeoDataFrame._from_data(data)
