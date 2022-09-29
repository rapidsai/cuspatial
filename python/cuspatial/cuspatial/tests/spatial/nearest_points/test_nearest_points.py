import geopandas as gpd
import pytest
import shapely
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point

import cuspatial


def compute_point_linestring_nearest_point_host(gs1, gs2):
    gpdf = gpd.GeoDataFrame({"lhs": gs1, "rhs": gs2})
    nearest_points = gpdf.apply(
        lambda row: shapely.ops.nearest_points(row["lhs"], row["rhs"]),
        axis=1,
        result_type="expand",
    )

    # Compute point geometry id
    def point_idx_in_multipoint(row):
        point = row["nearest_points"]
        multipoint = row["points"]
        if not isinstance(multipoint, MultiPoint):
            return 0

        for i, g in enumerate(multipoint.geoms):
            if g.equals(point):
                return i
        return -1

    points_gpdf = gpd.GeoDataFrame(
        {"nearest_points": nearest_points[0], "points": gs1}
    )
    point_geometry_id = points_gpdf.apply(point_idx_in_multipoint, axis=1)

    # compute linestring geometry id and segment id
    # shapely's predicate between point and linestring is too strict,
    # thus this resorts to a custom approxmated method via
    # `approximate_point_on_line`
    def point_idx_in_multilinestring(row):
        multilinestring = row["lines"]
        point = row["nearest_points"]
        for i, line in enumerate(multilinestring.geoms):
            for j, segment in enumerate(
                zip(line.coords[:-1], line.coords[1:])
            ):
                if LineString(segment).intersects(point.buffer(1e-8)):
                    return [i, j]
        return [-1, -1]

    lines_gpdf = gpd.GeoDataFrame(
        {"nearest_points": nearest_points[1], "lines": gs2}
    )
    line_geometry_id = lines_gpdf.apply(
        point_idx_in_multilinestring, axis=1, result_type="expand"
    )

    return gpd.GeoDataFrame(
        {
            "point_geometry_id": point_geometry_id,
            "linestring_geometry_id": line_geometry_id[0],
            "segment_id": line_geometry_id[1],
            "geometry": nearest_points[1],
        }
    )


def test_empty_input():
    points = cuspatial.GeoSeries([])
    linestrings = cuspatial.GeoSeries([])

    result = cuspatial.pairwise_point_linestring_nearest_points(
        points, linestrings
    )
    expected = gpd.GeoDataFrame(
        {
            "point_geometry_id": [],
            "linestring_geometry_id": [],
            "segment_id": [],
            "geometry": gpd.GeoSeries(),
        }
    )
    assert_geodataframe_equal(
        result.to_pandas(), expected, check_index_type=False
    )


def test_single():
    points = cuspatial.GeoSeries([Point(0, 0)])
    linestrings = cuspatial.GeoSeries([LineString([(2, 2), (1, 1)])])

    result = cuspatial.pairwise_point_linestring_nearest_points(
        points, linestrings
    )
    expected = gpd.GeoDataFrame(
        {
            "point_geometry_id": [0],
            "linestring_geometry_id": [0],
            "segment_id": [0],
            "geometry": gpd.GeoSeries([Point(1, 1)]),
        }
    )
    assert_geodataframe_equal(
        result.to_pandas(), expected, check_dtype=False, check_index_type=False
    )


def test_multipoints():
    points = cuspatial.GeoSeries(
        [MultiPoint([(0, 0), (0.5, 0.5), (1.0, 0.5)])]
    )
    linestrings = cuspatial.GeoSeries([LineString([(2, 2), (1, 1)])])

    result = cuspatial.pairwise_point_linestring_nearest_points(
        points, linestrings
    )
    expected = gpd.GeoDataFrame(
        {
            "point_geometry_id": [2],
            "linestring_geometry_id": [0],
            "segment_id": [0],
            "geometry": gpd.GeoSeries([Point(1, 1)]),
        }
    )
    assert_geodataframe_equal(
        result.to_pandas(), expected, check_dtype=False, check_index_type=False
    )


def test_multilinestrings():
    points = cuspatial.GeoSeries([Point(0.5, 0.5)])
    linestrings = cuspatial.GeoSeries(
        [
            MultiLineString(
                [
                    [(2, 2), (1, 1)],
                    [(1, 1), (2, 0)],
                    [(2, 0), (3, 1)],
                    [(3, 1), (2, 2)],
                ]
            )
        ]
    )

    result = cuspatial.pairwise_point_linestring_nearest_points(
        points, linestrings
    )
    expected = gpd.GeoDataFrame(
        {
            "point_geometry_id": [0],
            "linestring_geometry_id": [0],
            "segment_id": [0],
            "geometry": gpd.GeoSeries([Point(1, 1)]),
        }
    )
    assert_geodataframe_equal(
        result.to_pandas(), expected, check_dtype=False, check_index_type=False
    )


def test_random_input_nearest_point(
    multipoint_generator, multilinestring_generator
):
    num_pairs = 10
    max_num_points_per_multipoint = 10
    max_num_linestring_per_multilinestring = 10
    max_num_segments_per_linestring = 50

    points = gpd.GeoSeries(
        [*multipoint_generator(num_pairs, max_num_points_per_multipoint)]
    )
    linestrings = gpd.GeoSeries(
        [
            *multilinestring_generator(
                num_pairs,
                max_num_linestring_per_multilinestring,
                max_num_segments_per_linestring,
            )
        ]
    )

    d_points = cuspatial.from_geopandas(points)
    d_linestrings = cuspatial.from_geopandas(linestrings)

    result = cuspatial.pairwise_point_linestring_nearest_points(
        d_points, d_linestrings
    )
    expected = compute_point_linestring_nearest_point_host(points, linestrings)

    assert_geodataframe_equal(
        result.to_pandas(),
        expected,
        check_dtype=False,
        check_index_type=False,
        check_less_precise=True,
    )


def test_input_length_unequal():
    d_points = cuspatial.GeoSeries([Point(1, 1), Point(0, 0)])
    d_lines = cuspatial.GeoSeries(
        [
            MultiLineString(
                [
                    [(1, 2), (3, 4)],
                    [(1, 2), (3, 4)],
                ]
            )
        ]
    )

    with pytest.raises(
        ValueError, match="should have the same number of geometries"
    ):
        cuspatial.pairwise_point_linestring_nearest_points(d_points, d_lines)


def test_contains_mixed_geometries():
    d_point_lines = cuspatial.GeoSeries(
        [Point(1, 1), MultiLineString([[(1, 2), (3, 4)]])]
    )
    d_lines = cuspatial.GeoSeries(
        [
            MultiLineString([[(1, 2), (3, 4)], [(1, 2), (3, 4)]]),
            MultiLineString([[(5, 6), (7, 8)], [(5, 6), (7, 8)]]),
        ]
    )

    with pytest.raises(ValueError, match="must contain only point geometries"):
        cuspatial.pairwise_point_linestring_nearest_points(
            d_point_lines, d_lines
        )


def test_contains_mixed_geometries2():
    d_points = cuspatial.GeoSeries([Point(1, 1), Point(2, 2)])
    d_lines_point = cuspatial.GeoSeries(
        [MultiLineString([[(1, 2), (3, 4)], [(1, 2), (3, 4)]]), Point(5, 6)]
    )

    with pytest.raises(
        ValueError, match="must contain only linestring geometries"
    ):
        cuspatial.pairwise_point_linestring_nearest_points(
            d_points, d_lines_point
        )
