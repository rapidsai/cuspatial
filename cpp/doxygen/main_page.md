libcuspatial is a GPU-accelerated C++ library for spatial data analysis including distance and 
trajectory computations, spatial data indexing and spatial join operations. libcuspatial is 
the high-performance backend for the cuSpatial Python library.

libcuspatial has two interfaces. The generic header-only C++ API represents data as arrays
of structures (e.g. 2D points). The header-only API uses iterators for input and output, and is
similar in style to the C++ Standard Template Library (STL) and Thrust. All cuSpatial algorithms
are implemented in this API.

The libcuspatial "column-based API" is a C++ API based on data types from libcudf, 
[the CUDA Dataframe library C++ API](https://docs.rapids.ai/api/libcudf/nightly/index.html). The
column-based API represents spatial data as cuDF tables of type-erased columns, and layers on top
of the header-only API.

## Useful Links

 - [cuSpatial Github Repository](https://github.com/rapidsai/cuspatial)
 - [cuSpatial C++ Developer Guide](DEVELOPER_GUIDE.html)
 - [cuSpatial Python API Documentation](https://docs.rapids.ai/api/cuspatial/stable/)
 - [cuSpatial Python Developer Guide](https://docs.rapids.ai/api/cuspatial/stable/developer_guide/index.html)]
 - [RAPIDS Home Page](https://rapids.ai)
