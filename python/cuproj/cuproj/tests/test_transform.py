
from pyproj import Transformer

from numpy.testing import assert_allclose

from cuproj import Transformer as cuTransformer

def test_wgs84_to_utm_one_point():
    # Sydney opera house latitude and longitude
    lat = -33.8587
    lon = 151.2140

    # Transform to UTM using PyProj
    transformer = Transformer.from_crs("epsg:4326", "epsg:32756")
    pyproj_x, pyproj_y = transformer.transform(lat, lon)

    # Transform to UTM using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:4326", "epsg:32756")
    cuproj_x, cuproj_y = cu_transformer.transform(lat, lon)

    assert_allclose(cuproj_x, pyproj_x)
    assert_allclose(cuproj_y, pyproj_y)
