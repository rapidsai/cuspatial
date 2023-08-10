# cuProj: GPU-Accelerated Cartographic Projections and Coordinate Transformations

cuProj is a generic coordinate transformation library that transforms geospatial coordinates from
one coordinate reference system (CRS) to another. This includes cartographic projections as well as
geodetic transformations. cuProj is implemented in CUDA C++ to run on GPUs to provide the highest
performance.

cuProj provides a Python API that closely matches the
[PyProj](https://pyproj4.github.io/pyproj/stable/) API.

Currently cuProj only supports a subset of the Proj transformations. The following transformations are supported:

- WGS84 to/from UTM


```{toctree}
:maxdepth: 2
:caption: Contents

user_guide/index
api_docs/index
developer_guide/index
```

# Indices and tables

- {ref}`genindex`
- {ref}`search`
