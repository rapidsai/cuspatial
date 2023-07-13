
from pyproj import Transformer

from numpy.testing import assert_allclose

from cuproj import Transformer as cuTransformer

def test_wgs84_to_utm_one_point():
    # Sydney opera house latitude and longitude
    lat = -33.8587
    lon = 151.2140

    # Sydney opera house UTM zone 56S coordinates
    expected_x = 334783.95448071
    expected_y = 6252075.96174145

    # Transform to UTM using PyProj
    transformer = Transformer.from_crs("epsg:4326", "epsg:32756")
    pyproj_x, pyproj_y = transformer.transform(lat, lon)

    assert_allclose(pyproj_x, expected_x, atol=1)
    assert_allclose(pyproj_y, expected_y, atol=1)

    # Transform to UTM using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:4326", "epsg:32756")
    cuproj_x, cuproj_y = cu_transformer.transform(lat, lon)
