import pytest

from shapely.geometry import *

from cudf.testing import assert_series_equal

import cuspatial

@pytest.mark.parametrize(
    "lobj", [
        Point(0, 0),
        MultiPoint([
            Point(1, 1),
            Point(0, 0)
        ]),
        LineString([(0, 0), (1, 1)]),
        MultiLineString([
            LineString([(0, 0), (1, 1)]),
            LineString([(2, 2), (3, 3)]),
        ]),
        Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
        MultiPolygon([
            Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
            Polygon([(2, 2), (2, 3), (3, 2), (2, 2)])
        ])
    ]
)
@pytest.mark.parametrize(
    "robj", [
        Point(10, 10),
        MultiPoint([
            Point(11, 11),
            Point(10, 10)
        ]),
        LineString([(10, 10), (11, 11)]),
        MultiLineString([
            LineString([(10, 10), (11, 11)]),
            LineString([(12, 12), (13, 13)]),
        ]),
        Polygon([(10, 0), (11, 10), (10, 11), (10, 10)]),
        MultiPolygon([
            Polygon([(10, 10), (11, 10), (10, 11), (10, 10)]),
            Polygon([(12, 12), (12, 13), (13, 12), (12, 12)])
        ])
    ]
)
def test_one_pair_all_combos_no_nulls_test(lobj, robj):
    lhs = cuspatial.GeoSeries([lobj])
    rhs = cuspatial.GeoSeries([robj])

    hlhs = lhs.to_geopandas()
    hrhs = rhs.to_geopandas()

    got = lhs.distance(rhs)
    expect = hlhs.distance(hrhs)

    assert_series_equal(cudf.Series(expect), got)
