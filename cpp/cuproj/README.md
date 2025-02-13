# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuProj: GPU-Accelerated Coordinate Projection</div>

## Overview

cuProj is a generic coordinate transformation library that transforms geospatial coordinates from
one coordinate reference system (CRS) to another. This includes cartographic projections as well as
geodetic transformations. cuProj is implemented in CUDA C++ to run on GPUs to provide the highest
performance.

libcuproj is a CUDA C++ library that provides the header-only C++ API for cuProj. It is designed
to implement coordinate projections and transforms compatible with the [Proj](https://proj.org/)
library. The C++ API does not match the API of Proj, but it is designed to eventually expand to
support many of the same features and transformations that Proj supports.

Currently libcuproj only supports a subset of the Proj transformations. The following
transformations are supported:

- WGS84 to/from UTM

## Example

The C++ API is designed to be easy to use. The following example shows how to transform a point in
Sydney, Australia from WGS84 (lat, lon) coordinates to UTM zone 56S (x, y) coordinates.

```cpp
#include <cuproj/projection_factories.cuh>
#include <cuproj/vec_2d.hpp>

using T = float;

// Make a projection to convert WGS84 (lat, lon) coordinates to UTM zone 56S (x, y) coordinates
auto proj = cuproj::make_projection<cuproj::vec_2d<T>>("EPSG:4326", "EPSG:32756");

cuproj::vec_2d<T> sydney{-33.858700, 151.214000};  // Sydney, NSW, Australia
thrust::device_vector<cuproj::vec_2d<T>> d_in{1, sydney};
thrust::device_vector<cuproj::vec_2d<T>> d_out(d_in.size());

// Convert the coordinates. Works the same with a vector of many coordinates.
proj.transform(d_in.begin(), d_in.end(), d_out.begin(), cuproj::direction::FORWARD);
```

### Projections in CUDA device code

The C++ API also supports transforming coordinate in CUDA device code. Create a
`projection` as above, then get a `device_projection` object from it, which can
be passed to a kernel launch. Here's an example kernel.

```cpp
using device_projection = cuproj::device_projection<cuproj::vec_2d<float>>;

__global__ void example_kernel(device_projection const d_proj,
                               cuproj::vec_2d<float> const* in,
                               cuproj::vec_2d<float>* out,
                               size_t n)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < n;
       i += gridDim.x * blockDim.x) {
    out[i] = d_proj.transform(in[i]);
  }
}
```

The corresponding host code:

```cpp
using coordinate = cuproj::vec_2d<float>;

// Make a projection to convert WGS84 (lat, lon) coordinates to
// UTM zone 56S (x, y) coordinates
auto proj = cuproj::make_projection<coordinate>("EPSG:4326", "EPSG:32756");

// Sydney, NSW, Australia
coordinate sydney{-33.858700, 151.214000};
thrust::device_vector<coordinate> d_in{1, sydney};
thrust::device_vector<coordinate> d_out(d_in.size());

auto d_proj            = proj->get_device_projection(cuproj::direction::FORWARD);
std::size_t block_size = 256;
std::size_t grid_size  = (d_in.size() + block_size - 1) / block_size;
example_kernel<<<grid_size, block_size>>>(
  d_proj, d_in.data().get(), d_out.data().get(), d_in.size());
cudaDeviceSynchronize();
```