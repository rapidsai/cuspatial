# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuSpatial - GPU-Accelerated Vector Geospatial Data Analysis</div>

> **Note**
>
> cuSpatial depends on [RMM](https://github.com/rapidsai/rmm) from [RAPIDS](https://rapids.ai/).

## Resources

## Overview

cuProj is a generic coordinate transformation library that transforms geospatial coordinates from
one coordinate reference system (CRS) to another. This includes cartographic projections as well as geodetic transformations. cuProj is implemented in CUDA C++ to run on GPUs to provide the highest performance.

cuProj provides a Python API that closely matches the [PyProj](https://pyproj4.github.io/pyproj/stable/) API. cuProj also provides a header-only C++ API. While the C++ API does not match the API
of [Proj](https://proj.org/), it is designed to eventually expand to support many of the same features
and transformations that Proj supports.

Currently cuProj only supports a subset of the Proj transformations. The following transformations are supported:

- WGS84 to/from UTM

## Example

The Python API is closely matched to PyProj and data can seamlessly transition between the two:

```python
import cuproj
import pyproj

# Create a PyProj transformer
pyproj_transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32613")

# Create a cuProj transformer
cuproj_transformer = cuproj.Transformer.from_crs("EPSG:4326", "EPSG:32613")

# Transform a grid of points around the San Francisco Bay using PyProj
num_points = 10000
grid_side = int(np.sqrt(num_points))

x, y = np.meshgrid(np.linspace(min_corner[0], max_corner[0], grid_side),
                   np.linspace(min_corner[1], max_corner[1], grid_side))
grid = [x.reshape(-1), y.reshape(-1)]

pyproj_result = pyproj_transformer.transform(*grid)

# Transform a grid of points around the San Francisco Bay using cuProj
cuproj_result = cuproj_transformer.transform(*grid)
```

Note that the cuProj transformer is created from the same CRSs as the PyProj transformer. The
transformer can then be used to transform a grid of points. The result of the transformation is
returned as a tuple of x and y coordinates. The result of the PyProj transformation is a tuple of
Numpy arrays, while the result of the cuProj transformation is a tuple of
[CuPy](https://cupy.dev/) arrays.

Also note that in the above example, the input data are in host memory, so cuProj will create a
copy in device memory first. Data already on the device will not be copied, resulting in higher
performance. See the
[simple cuProj Benchmark notebook](../../notebooks/simple_cuproj_benchmark.ipynb) for an example.
