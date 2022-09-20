import cupy as cp

import cuspatial._lib.nearest_points as nearest_points
from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial import GeoDataFrame, GeoSeries
from cuspatial.utils import contains_only_linestrings, contains_only_points

def pairwise_point_linestring_nearest_points(
    points: GeoSeries, linestrings: GeoSeries
) -> GeoDataFrame:
    """
    Returns the nearest points between two GeoSeries of points and linestrings.

    Parameters
    ----------
    points : GeoSeries
        A GeoSeries of points. Currently either only a series of points or
        a series of multi-points is supported.
    linestrings : GeoSeries
        A GeoSeries of linestrings.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with four columns. 
        - "point_geometry_id" indicates the index of the nearest point in the
          `points` GeoSeries.
        - "linestring_geometry_id" indicates the index of the linestring where
          the nearest point is located.
        - "segment_id" indicates the index of the segment where the nearest
          point is located.
        - "nearest_point_on_linestring" contains the points of the nearest
          point on the linestring.
    """

    if not contains_only_points(GeoSeries):
        raise ValueError("`points` must contain only point geometries.")

    if not contains_only_linestrings(GeoSeries):
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
        else points.multipoints.geometry_offset._column
    )

    (
        point_geometry_id,
        linestring_geometry_id,
        segment_id,
        point_on_linestring_xy,
    ) = nearest_points.pairwise_point_linestring_nearest_points(
        points_xy,
        linestrings.lines.part_offsets._column,
        linestrings.lines.points._column,
        points_geometry_offset,
        linestrings.lines.geometry_offset._column,
    )

    point_on_linestring = GeoColumn._from_points_xy(point_on_linestring_xy)
    nearest_points_on_linestring = GeoSeries(point_on_linestring)

    data = {}
    if not point_geometry_id:
        point_geometry_id = cp.zeros(len(points), dtype=cp.int32)

    data.update(
        {
            "point_geometry_id": point_geometry_id,
            "linestring_geometry_id": linestring_geometry_id,
            "segment_id": segment_id,
            "nearest_point_on_linestring": nearest_points_on_linestring,
        }
    )

    # TODO
    return GeoDataFrame._from_data(data)
