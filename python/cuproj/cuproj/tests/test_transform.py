
from numpy.testing import assert_allclose
from pyproj import Transformer

from cuproj import Transformer as cuTransformer


def test_wgs84_to_utm_one_point():
    # Sydney opera house latitude and longitude
    lat = -33.8587
    lon = 151.2140

    # Transform to UTM using PyProj
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32756")
    pyproj_x, pyproj_y = transformer.transform(lat, lon)

    # Transform to UTM using cuproj
    cu_transformer = cuTransformer.from_crs("EPSG:4326", "EPSG:32756")
    cuproj_x, cuproj_y = cu_transformer.transform(lat, lon)

    assert_allclose(cuproj_x, pyproj_x)
    assert_allclose(cuproj_y, pyproj_y)

# def grid_generator(min_corner, max_corner, num_points_x, num_points_y):
#     spacing = ((max_corner[0] - min_corner[0]) / num_points_x,
#                (max_corner[1] - min_corner[1]) / num_points_y)
#     for i in range(num_points_x * num_points_y):
#         yield (min_corner[0] + (i % num_points_x) * spacing[0],
#                min_corner[1] + (i // num_points_x) * spacing[1])

# # test with a grid of points around san francisco
# def test_wgs84_to_utm_grid():
#     # San Francisco bounding box
#     min_corner = (-122.5149, 37.7081)
#     max_corner = (-122.3573, 37.8324)

#     # Define the number of points in the grid
#     num_points_x = 100
#     num_points_y = 100

#     # Transform to UTM using PyProj
#     transformer = Transformer.from_crs(
#         "EPSG:4326", "EPSG:32610", always_xy=True)
#     pyproj_x, pyproj_y = transformer.transform(
#         *zip(*grid_generator(min_corner, max_corner, 100, 100)))

#     # Transform to UTM using cuproj
#     cu_transformer = cuTransformer.from_crs("EPSG:4326", "EPSG:32610")
#     cuproj_x, cuproj_y = cu_transformer.transform(
#         *zip(*grid_generator(min_corner, max_corner, 100, 100)))

#     assert_allclose(cuproj_x, pyproj_x)
#     assert_allclose(cuproj_y, pyproj_y)
