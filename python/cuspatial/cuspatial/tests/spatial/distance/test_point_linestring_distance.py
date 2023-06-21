import geopandas as gpd
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)

import cudf

from cuspatial import pairwise_point_linestring_distance
from cuspatial.io.geopandas import from_geopandas


@pytest.mark.parametrize(
    "points, lines",
    [
        ([Point(0, 0)], [LineString([(1, 0), (0, 1)])]),
        ([MultiPoint([(0, 0), (0.5, 0.5)])], [LineString([(1, 0), (0, 1)])]),
        (
            [Point(0, 0)],
            [MultiLineString([[(1, 0), (0, 1)], [(0, 0), (1, 1)]])],
        ),
        (
            [MultiPoint([(0, 0), (1, 1)])],
            [MultiLineString([[(1, 0), (0, 1)], [(0, 0), (1, 1)]])],
        ),
    ],
)
def test_one_pair(points, lines):
    hpts = gpd.GeoSeries(points)
    hls = gpd.GeoSeries(lines)

    gpts = from_geopandas(hpts)
    gls = from_geopandas(hls)

    got = pairwise_point_linestring_distance(gpts, gls)
    expected = hpts.distance(hls)

    cudf.testing.assert_series_equal(got, cudf.Series(expected))


@pytest.mark.parametrize("num_pairs", [1, 2, 100])
def test_random_point_linestring_pairs(
    num_pairs, point_generator, linestring_generator
):
    max_num_segments_per_linestring = 50

    hpts = gpd.GeoSeries([*point_generator(num_pairs)])
    hlines = gpd.GeoSeries(
        [*linestring_generator(num_pairs, max_num_segments_per_linestring)]
    )

    gpts = from_geopandas(hpts)
    glines = from_geopandas(hlines)

    got = pairwise_point_linestring_distance(gpts, glines)
    expected = hpts.distance(hlines)

    cudf.testing.assert_series_equal(got, cudf.Series(expected))


@pytest.mark.parametrize("num_pairs", [1, 2, 100])
def test_random_multipoint_linestring_pairs(
    num_pairs, multipoint_generator, linestring_generator
):
    max_num_points_per_multipoint = 10
    max_num_segments_per_linestring = 50

    hpts = gpd.GeoSeries(
        [*multipoint_generator(num_pairs, max_num_points_per_multipoint)]
    )
    hlines = gpd.GeoSeries(
        [*linestring_generator(num_pairs, max_num_segments_per_linestring)]
    )

    gpts = from_geopandas(hpts)
    glines = from_geopandas(hlines)

    got = pairwise_point_linestring_distance(gpts, glines)
    expected = hpts.distance(hlines)

    cudf.testing.assert_series_equal(got, cudf.Series(expected))


@pytest.mark.parametrize("num_pairs", [1, 2, 100])
def test_random_point_multilinestring_pairs(
    num_pairs, point_generator, multilinestring_generator
):
    max_num_linestring_per_multilinestring = 10
    max_num_segments_per_linestring = 50

    hpts = gpd.GeoSeries([*point_generator(num_pairs)])
    hlines = gpd.GeoSeries(
        [
            *multilinestring_generator(
                num_pairs,
                max_num_linestring_per_multilinestring,
                max_num_segments_per_linestring,
            )
        ]
    )

    gpts = from_geopandas(hpts)
    glines = from_geopandas(hlines)

    got = pairwise_point_linestring_distance(gpts, glines)
    expected = hpts.distance(hlines)

    cudf.testing.assert_series_equal(got, cudf.Series(expected))


@pytest.mark.parametrize("num_pairs", [1, 2, 100])
def test_random_multipoint_multilinestring_pairs(
    num_pairs, multipoint_generator, multilinestring_generator
):
    max_num_points_per_multipoint = 10
    max_num_linestring_per_multilinestring = 10
    max_num_segments_per_linestring = 50

    hpts = gpd.GeoSeries(
        [*multipoint_generator(num_pairs, max_num_points_per_multipoint)]
    )
    hlines = gpd.GeoSeries(
        [
            *multilinestring_generator(
                num_pairs,
                max_num_linestring_per_multilinestring,
                max_num_segments_per_linestring,
            )
        ]
    )

    gpts = from_geopandas(hpts)
    glines = from_geopandas(hlines)

    got = pairwise_point_linestring_distance(gpts, glines)
    expected = hpts.distance(hlines)

    cudf.testing.assert_series_equal(got, cudf.Series(expected))


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (
            [Point(0, 0), LineString([(1.0, 1.0), (1.0, 2.0)])],
            [
                LineString([(0.5, 0.5), (1.2, 1.7)]),
                LineString([(8.0, 6.4), (11.3, 21.7)]),
            ],
        ),
        (
            [Point(1.0, 2.0), Point(1.0, 1.0)],
            [
                LineString([(0.5, 0.5), (1.2, 1.7)]),
                Polygon([(8.0, 6.4), (0.0, 0.0), (-5.0, -7.8)]),
            ],
        ),
    ],
)
def test_mixed_geometry_series_raise(lhs, rhs):
    lhs = from_geopandas(gpd.GeoSeries(lhs))
    rhs = from_geopandas(gpd.GeoSeries(rhs))

    with pytest.raises(ValueError, match=".*must contain only.*"):
        pairwise_point_linestring_distance(lhs, rhs)


def test_multipoint_multilinestring_sliced_pairs(
    multipoint_generator, multilinestring_generator
):
    num_pairs = 2
    max_num_linestring_per_multilinestring = 2
    max_num_segments_per_linestring = 2
    max_num_points_per_multipoint = 2

    hpts = gpd.GeoSeries(
        [*multipoint_generator(num_pairs, max_num_points_per_multipoint)]
    )
    hlines = gpd.GeoSeries(
        [
            *multilinestring_generator(
                num_pairs,
                max_num_linestring_per_multilinestring,
                max_num_segments_per_linestring,
            )
        ]
    )

    gpts = from_geopandas(hpts)
    glines = from_geopandas(hlines)

    hslicepts = hpts[0:1]
    hslicelines = hlines[0:1]
    slicegpts = gpts[0:1]
    sliceglines = glines[0:1]
    got = pairwise_point_linestring_distance(slicegpts, sliceglines)
    expected = hslicepts.distance(hslicelines)

    cudf.testing.assert_series_equal(got, cudf.Series(expected))

    hslicepts = hpts[1:2]
    hslicelines = hlines[1:2]
    slicegpts = gpts[1:2]
    sliceglines = glines[1:2]
    got = pairwise_point_linestring_distance(slicegpts, sliceglines)
    expected = hslicepts.distance(hslicelines)

    # Just dropping the index for now, that is not libcuSpatial's problem
    got.index = [1]
    cudf.testing.assert_series_equal(got, cudf.Series(expected))


@pytest.mark.parametrize("slice_index", [0, 1, 2, 3, 4])
def test_multipoint_multilinestring_sliced_many(
    multipoint_generator, multilinestring_generator, slice_twenty, slice_index
):
    num_pairs = 20
    max_num_linestring_per_multilinestring = 5
    max_num_segments_per_linestring = 5
    max_num_points_per_multipoint = 5

    hpts = gpd.GeoSeries(
        [*multipoint_generator(num_pairs, max_num_points_per_multipoint)]
    )
    hlines = gpd.GeoSeries(
        [
            *multilinestring_generator(
                num_pairs,
                max_num_linestring_per_multilinestring,
                max_num_segments_per_linestring,
            )
        ]
    )

    gpts = from_geopandas(hpts)
    glines = from_geopandas(hlines)

    pslice = slice_twenty[slice_index]
    hslicepts = hpts[pslice]
    hslicelines = hlines[pslice]
    slicegpts = gpts[pslice]
    sliceglines = glines[pslice]
    got = pairwise_point_linestring_distance(slicegpts, sliceglines)
    expected = hslicepts.distance(hslicelines)
    got.index = cudf.RangeIndex(pslice.start, pslice.stop)

    cudf.testing.assert_series_equal(got, cudf.Series(expected))


def test_multipoint_multilinestring_sliced_noncontiguous(
    multipoint_generator, multilinestring_generator
):
    num_pairs = 20
    max_num_linestring_per_multilinestring = 5
    max_num_segments_per_linestring = 5
    max_num_points_per_multipoint = 5

    hpts = gpd.GeoSeries(
        [*multipoint_generator(num_pairs, max_num_points_per_multipoint)]
    )
    hlines = gpd.GeoSeries(
        [
            *multilinestring_generator(
                num_pairs,
                max_num_linestring_per_multilinestring,
                max_num_segments_per_linestring,
            )
        ]
    )

    gpts = from_geopandas(hpts)
    glines = from_geopandas(hlines)

    pslice = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    hslicepts = hpts[pslice]
    hslicelines = hlines[pslice]
    slicegpts = gpts[pslice]
    sliceglines = glines[pslice]
    got = pairwise_point_linestring_distance(slicegpts, sliceglines)
    expected = hslicepts.distance(hslicelines)
    got.index = pslice

    cudf.testing.assert_series_equal(got, cudf.Series(expected))
