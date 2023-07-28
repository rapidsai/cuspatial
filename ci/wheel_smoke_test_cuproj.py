# Copyright (c) 2023, NVIDIA CORPORATION.

from cuproj import Transformer as cuTransformer

from cupy.testing import assert_allclose

if __name__ == '__main__':
    # Sydney opera house latitude and longitude
    lat = -33.8587
    lon = 151.2140

    # Transform to UTM using cuproj
    cu_transformer = cuTransformer.from_crs("epsg:4326", "EPSG:32756")
    cuproj_x, cuproj_y = cu_transformer.transform(lat, lon)

    assert_allclose(cuproj_x, 334783.9544807102)
    assert_allclose(cuproj_y, 6252075.961741454)
