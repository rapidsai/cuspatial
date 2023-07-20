
import pytest
from cupy.testing import assert_allclose
from pyproj import Transformer

from cuproj import Transformer as cuTransformer

valid_crs_combos = [
    (4326, 32756),
    (32756, 4326),
    (4326, 32610),
    (32610, 4326)]

invalid_crs_combos = [
    (4326, 4326),
    (32756, 32756),
    (4326, 756),
    (756, 4326)]


def to_epsg_string(code):
    return f"epsg:{code}"


@pytest.mark.parametrize("crs_from, crs_to", valid_crs_combos)
def test_valid_epsg_codes(crs_from, crs_to):
    Transformer.from_crs(crs_from, crs_to)


@pytest.mark.parametrize("crs_from, crs_to", valid_crs_combos)
def test_valid_epsg_strings(crs_from, crs_to):
    Transformer.from_crs(to_epsg_string(crs_from), to_epsg_string(crs_to))


@pytest.mark.parametrize("crs_from, crs_to", valid_crs_combos)
def test_valid_uppercase_epsg_strings(crs_from, crs_to):
    Transformer.from_crs(
        to_epsg_string(crs_from).upper(), to_epsg_string(crs_to).upper())


@pytest.mark.parametrize("crs_from, crs_to", invalid_crs_combos)
def test_invalid_epsg_codes(crs_from, crs_to):
    with pytest.raises(RuntimeError):
        cuTransformer.from_crs(crs_from, crs_to)


@pytest.mark.parametrize("crs_from, crs_to", invalid_crs_combos)
def test_invalid_epsg_strings(crs_from, crs_to):
    with pytest.raises(RuntimeError):
        cuTransformer.from_crs(
            to_epsg_string(crs_from), to_epsg_string(crs_to))


def test_wgs84_to_utm_one_point():
    # Sydney opera house latitude and longitude
    lat = -33.8587
    lon = 151.2140

    # Transform to UTM using PyProj
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32756")
    pyproj_x, pyproj_y = transformer.transform(lat, lon)

    # Transform to UTM using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:4326", "EPSG:32756")
    cuproj_x, cuproj_y = cu_transformer.transform(lat, lon)

    assert_allclose(cuproj_x, pyproj_x)
    assert_allclose(cuproj_y, pyproj_y)


def grid_generator(min_corner, max_corner, num_points_lat, num_points_lon):
    spacing = ((max_corner[0] - min_corner[0]) / num_points_lat,
               (max_corner[1] - min_corner[1]) / num_points_lon)
    for i in range(num_points_lat * num_points_lon):
        yield (min_corner[0] + (i % num_points_lon) * spacing[0],
               min_corner[1] + (i // num_points_lon) * spacing[1])


# test with a grid of points around san francisco
def test_wgs84_to_utm_grid():
    # San Francisco bounding box
    min_corner = (37.7081, -122.5149)
    max_corner = (37.8324, -122.3573)

    # Define the number of points in the grid
    num_points_x = 2
    num_points_y = 2

    # Transform to UTM using PyProj
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32756")
    pyproj_x, pyproj_y = transformer.transform(
        *zip(*grid_generator(
            min_corner, max_corner, num_points_x, num_points_y)))

    print(f"pyproj_x: {pyproj_x}")
    print(f"pyproj_y: {pyproj_y}")

    # Transform to UTM using cuproj
    cu_transformer = cuTransformer.from_crs("EPSG:4326", "EPSG:32756")
    cuproj_x, cuproj_y = cu_transformer.transform(
        *zip(*grid_generator(
            min_corner, max_corner, num_points_x, num_points_y)))

    assert_allclose(cuproj_x, pyproj_x)
    assert_allclose(cuproj_y, pyproj_y)
