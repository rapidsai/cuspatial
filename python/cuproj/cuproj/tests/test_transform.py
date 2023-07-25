
import cupy as cp
import geopandas as gpd
import numpy as np
import pytest
from cupy.testing import assert_allclose
from pyproj import Transformer
from pyproj.enums import TransformDirection
from shapely.geometry import Point

import cuspatial
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


@pytest.mark.parametrize("crs_from, crs_to", valid_crs_combos)
def test_valid_epsg_mixed_ints_strings(crs_from, crs_to):
    Transformer.from_crs(to_epsg_string(crs_from), crs_to)
    Transformer.from_crs(str(crs_from), crs_to)
    Transformer.from_crs(crs_from, to_epsg_string(crs_to))
    Transformer.from_crs(crs_from, str(crs_to))


@pytest.mark.parametrize("crs_from, crs_to", valid_crs_combos)
def test_valid_epsg_tuples(crs_from, crs_to):
    Transformer.from_crs(("EPSG", crs_from), ("EPSG", crs_to))
    Transformer.from_crs(("EPSG", crs_from), crs_to)
    Transformer.from_crs(("epsg", crs_from), to_epsg_string(crs_to))
    Transformer.from_crs(("EPSG", crs_from), str(crs_to))
    with pytest.raises(RuntimeError):
        Transformer.from_crs(("RPG", crs_from), crs_to)  # invalid authority


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
        dtype, container_type, min_corner, max_corner, crs_to):
    num_points_x = 100
    num_points_y = 100

    x, y = np.meshgrid(
        np.linspace(min_corner[0], max_corner[0], num_points_y, dtype=dtype),
        np.linspace(min_corner[1], max_corner[1], num_points_x, dtype=dtype))
    grid = [x.reshape(-1), y.reshape(-1)]

    # Transform to UTM using PyProj
    transformer = Transformer.from_crs("EPSG:4326", crs_to)
    pyproj_x, pyproj_y = transformer.transform(*grid)

    # Transform to UTM using cuproj
    cu_grid = container_type(grid)
    cu_transformer = cuTransformer.from_crs("EPSG:4326", crs_to)
    cuproj_x, cuproj_y = cu_transformer.transform(*cu_grid)

    # Expect within 5m for float32, 100nm for float64
    atol = 5 if dtype == cp.float32 else 1e-07

    assert_allclose(cuproj_x, pyproj_x, atol=atol)
    assert_allclose(cuproj_y, pyproj_y, atol=atol)

    # Transform back to WGS84 using PyProj
    pyproj_x_back, pyproj_y_back = transformer.transform(
        pyproj_x, pyproj_y, direction=TransformDirection.INVERSE)

    # Transform back to WGS84 using cuproj
    cuproj_x_back, cuproj_y_back = cu_transformer.transform(
        cuproj_x, cuproj_y, direction="INVERSE")

    assert_allclose(cuproj_x_back, pyproj_x_back, atol=atol)
    assert_allclose(cuproj_y_back, pyproj_y_back, atol=atol)

    # Also test inverse-constructed Transformers

    # Transform back to WGS84 using PyProj
    transformer = Transformer.from_crs(crs_to, "EPSG:4326")
    pyproj_x_back, pyproj_y_back = transformer.transform(
        pyproj_x, pyproj_y)

    # Transform back to WGS84 using cuproj
    cu_transformer = cuTransformer.from_crs(crs_to, "EPSG:4326")
    cuproj_x_back, cuproj_y_back = cu_transformer.transform(
        cuproj_x, cuproj_y)

    assert_allclose(cuproj_x_back, pyproj_x_back, atol=atol)
    assert_allclose(cuproj_y_back, pyproj_y_back, atol=atol)


# test float and double
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
# test with grids of points
@pytest.mark.parametrize("container_type", container_types)
# test with various container types (host and device)
@pytest.mark.parametrize("min_corner, max_corner, crs_to", grid_corners)
def test_wgs84_to_utm_grid(dtype, container_type,
                           min_corner, max_corner, crs_to):
    run_forward_and_inverse_transforms(
        dtype, container_type, min_corner, max_corner, crs_to)


# test __cuda_array_interface__ support by using cuspatial geoseries as input
def test_geoseries_input():
    s = gpd.GeoSeries(
        [
            Point(grid_corners[0][0]),
            Point(grid_corners[0][1]),
        ]
    )

    gs = cuspatial.from_geopandas(s)

    # Transform to UTM using PyProj
    transformer = Transformer.from_crs("EPSG:4326", grid_corners[0][2])
    pyproj_x, pyproj_y = transformer.transform(s.x.values, s.y.values)

    # Transform to UTM using cuproj
    cu_transformer = cuTransformer.from_crs("EPSG:4326", grid_corners[0][2])
    cuproj_x, cuproj_y = cu_transformer.transform(gs.points.x, gs.points.y)

    assert_allclose(cuproj_x, pyproj_x)
    assert_allclose(cuproj_y, pyproj_y)
