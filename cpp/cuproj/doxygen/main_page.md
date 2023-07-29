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

// Make a projection to convert WGS84 (lat, lon) coordinates to UTM zone 56S (x, y) coordinates
auto proj = cuproj::make_projection<cuproj::vec_2d<T>>("EPSG:4326", "EPSG:32756");

cuproj::vec_2d<T> sydney{-33.858700, 151.214000};  // Sydney, NSW, Australia
thrust::device_vector<cuproj::vec_2d<T>> d_in{1, sydney};
thrust::device_vector<cuproj::vec_2d<T>> d_out(d_in.size());

// Convert the coordinates. Works the same with a vector of many coordinates.
proj.transform(d_in.begin(), d_in.end(), d_out.begin(), cuproj::direction::FORWARD);
```

## Useful Links

 - [RAPIDS Home Page](https://rapids.ai)
