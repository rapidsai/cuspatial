from shapely.geometry import LineString, Point, Polygon

import cuspatial

"""Overlaps, Within, and Intersects"""


def _test(lhs, rhs, predicate):
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()
    got = getattr(lhs, predicate)(rhs).values_host
    expected = getattr(gpdlhs, predicate)(gpdrhs).values
    assert (got == expected).all()


def test_polygon_overlaps_point():
    lhs = cuspatial.GeoSeries(
        [Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])]
    )
    rhs = cuspatial.GeoSeries([Point(0.5, 0.5)])
    _test(lhs, rhs, "overlaps")


def test_max_polygons_overlaps_max_points(polygon_generator, point_generator):
    lhs = cuspatial.GeoSeries([*polygon_generator(31, 0)])
    rhs = cuspatial.GeoSeries([*point_generator(31)])
    _test(lhs, rhs, "overlaps")


def test_polygon_overlaps_polygon_partially():
    lhs = cuspatial.GeoSeries(
        [Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])]
    )
    rhs = cuspatial.GeoSeries(
        [Polygon([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5], [0.5, 0.5]])]
    )
    _test(lhs, rhs, "overlaps")


def test_polygon_overlaps_polygon_completely():
    lhs = cuspatial.GeoSeries(
        [Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])]
    )
    rhs = cuspatial.GeoSeries(
        [
            Polygon(
                [
                    [0.25, 0.25],
                    [0.25, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.25],
                    [0.25, 0.25],
                ]
            )
        ]
    )
    _test(lhs, rhs, "overlaps")


def test_polygon_overlaps_polygon_no_overlap():
    lhs = cuspatial.GeoSeries(
        [Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])]
    )
    rhs = cuspatial.GeoSeries(
        [Polygon([[2, 2], [2, 3], [3, 3], [3, 2], [2, 2]])]
    )
    _test(lhs, rhs, "overlaps")


def test_max_polygon_overlaps_max_points(polygon_generator, point_generator):
    lhs = cuspatial.GeoSeries([*polygon_generator(31, 0)])
    rhs = cuspatial.GeoSeries([*point_generator(31)])
    _test(lhs, rhs, "overlaps")


def test_point_intersects_polygon_interior():
    lhs = cuspatial.GeoSeries([Point(0.5, 0.5)])
    rhs = cuspatial.GeoSeries(
        [Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])]
    )
    _test(lhs, rhs, "intersects")


def test_max_points_intersects_max_polygons_interior(
    polygon_generator, point_generator
):
    lhs = cuspatial.GeoSeries([*polygon_generator(31, 0)])
    rhs = cuspatial.GeoSeries([*point_generator(31)])
    _test(lhs, rhs, "intersects")


def test_point_within_polygon():
    lhs = cuspatial.GeoSeries([Point(0, 0)])
    rhs = cuspatial.GeoSeries([Polygon([[0, 0], [1, 0], [1, 1], [0, 0]])])
    _test(lhs, rhs, "within")


def test_max_points_within_max_polygons(polygon_generator, point_generator):
    lhs = cuspatial.GeoSeries([*polygon_generator(31, 0)])
    rhs = cuspatial.GeoSeries([*point_generator(31)])
    _test(lhs, rhs, "within")


def test_linestring_within_polygon():
    lhs = cuspatial.GeoSeries([LineString([(0, 0), (1, 1)])])
    rhs = cuspatial.GeoSeries([Polygon([[0, 0], [1, 0], [1, 1], [0, 0]])])
    _test(lhs, rhs, "within")


def test_max_linestring_within_max_polygon(
    polygon_generator, linestring_generator
):
    lhs = cuspatial.GeoSeries([*polygon_generator(31, 0)])
    rhs = cuspatial.GeoSeries([*linestring_generator(31, 5)])
    _test(lhs, rhs, "within")


def test_polygon_within_polygon():
    lhs = cuspatial.GeoSeries(
        [Polygon([[0, 0], [-1, 1], [1, 1], [1, -2], [0, 0]])]
    )
    rhs = cuspatial.GeoSeries([Polygon([[-1, -1], [-2, 2], [2, 2], [2, -2]])])
    _test(lhs, rhs, "within")


def test_max_polygons_within_max_polygons(polygon_generator):
    lhs = cuspatial.GeoSeries([*polygon_generator(31, 0)])
    rhs = cuspatial.GeoSeries([*polygon_generator(31, 1)])
    _test(lhs, rhs, "within")


def test_polygon_overlaps_linestring():
    lhs = cuspatial.GeoSeries(
        [Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])]
    )
    rhs = cuspatial.GeoSeries([LineString([(0.5, 0.5), (1.5, 1.5)])])
    _test(lhs, rhs, "overlaps")


def test_max_polygons_overlaps_max_linestrings(
    polygon_generator, linestring_generator
):
    lhs = cuspatial.GeoSeries([*polygon_generator(31, 0)])
    rhs = cuspatial.GeoSeries([*linestring_generator(31, 5)])
    _test(lhs, rhs, "overlaps")
