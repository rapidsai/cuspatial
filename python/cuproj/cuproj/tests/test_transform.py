
import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose
from pyproj import Transformer
from pyproj.enums import TransformDirection

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


grid_corners = [
    # San Francisco
    ((37.7081, -122.5149), (37.8324, -122.3573), "EPSG:32610"),
    # Sydney
    ((-33.9, 151.2), (-33.7, 151.3), "EPSG:32756"),
    # London
    ((51.3, -0.5), (51.6, 0.3), "EPSG:32630"),
    # New York City
    ((40.4774, -74.2591), (40.9176, -73.7004), "EPSG:32618"),
    # Ushuaia, Argentina
    ((-54.9, -68.4), (-54.7, -68.1), "EPSG:32719"),
    # McMurdo Station, Antarctica
    ((-77.9, 166.4), (-77.7, 166.7), "EPSG:32706"),
    # Singapore
    ((1.2, 103.6), (1.5, 104.0), "EPSG:32648")]

container_types = [list, tuple, np.asarray, cp.asarray]


def run_forward_and_inverse_transforms(
        container_type, min_corner, max_corner, crs_to):
    num_points_x = 100
    num_points_y = 100

    grid = np.meshgrid(np.linspace(min_corner[0], max_corner[0], num_points_y),
                       np.linspace(min_corner[1], max_corner[1], num_points_x))
    grid = [np.ravel(grid[0]), np.ravel(grid[1])]

    # Transform to UTM using PyProj
    transformer = Transformer.from_crs("EPSG:4326", crs_to)
    pyproj_x, pyproj_y = transformer.transform(*grid)

    # Transform to UTM using cuproj
    cu_grid = container_type(grid)
    cu_transformer = cuTransformer.from_crs("EPSG:4326", crs_to)
    cuproj_x, cuproj_y = cu_transformer.transform(*cu_grid)

    assert_allclose(cuproj_x, pyproj_x)
    assert_allclose(cuproj_y, pyproj_y)

    # Transform back to WGS84 using PyProj
    pyproj_x_back, pyproj_y_back = transformer.transform(
            pyproj_x, pyproj_y, direction=TransformDirection.INVERSE)

    # Transform back to WGS84 using cuproj
    cuproj_x_back, cuproj_y_back = cu_transformer.transform(
        cuproj_x, cuproj_y, direction="INVERSE")

    assert_allclose(cuproj_x_back, pyproj_x_back)
    assert_allclose(cuproj_y_back, pyproj_y_back)

    # Also test inverse-constructed Transformers

    # Transform back to WGS84 using PyProj
    transformer = Transformer.from_crs(crs_to, "EPSG:4326")
    pyproj_x_back, pyproj_y_back = transformer.transform(
            pyproj_x, pyproj_y)

    # Transform back to WGS84 using cuproj
    cu_transformer = cuTransformer.from_crs(crs_to, "EPSG:4326")
    cuproj_x_back, cuproj_y_back = cu_transformer.transform(
        cuproj_x, cuproj_y)

    assert_allclose(cuproj_x_back, pyproj_x_back)
    assert_allclose(cuproj_y_back, pyproj_y_back)

# test with grids of points
@pytest.mark.parametrize("container_type", container_types)
# test with various container types (host and device)
@pytest.mark.parametrize("min_corner, max_corner, crs_to", grid_corners)
def test_wgs84_to_utm_grid(container_type, min_corner, max_corner, crs_to):
    run_forward_and_inverse_transforms(
        container_type, min_corner, max_corner, crs_to)
