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

There are some basic examples of using the libcuproj C++ API in the 
[cuProj README](https://github.com/rapidsai/cuspatial/cpp/cuproj/README.md).

## Useful Links

 - [RAPIDS Home Page](https://rapids.ai)
 - [cuSpatial Github](https://github.com/rapidsai/cuspatial)
