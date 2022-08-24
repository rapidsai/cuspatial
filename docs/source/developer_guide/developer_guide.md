# cuSpatial Developer Guide

This documentation provides guidelines for developer on how to develop for cuspatial.
cuSpatial contains a GPU accelerated c++ library `libcuspatial` and a user friendly
python interface.
For guidelines on how to contribute to c++ API in libcuspatial,
see [cuSpatial C++ API Refactoring Guide](https://github.com/rapidsai/cuspatial/blob/main/cpp/doc/libcuspatial_refactoring_guide.md).

The APIs are wrapped with [cython](https://cython.readthedocs.io/en/latest/).
cuSpatial uses cython [`pxd` files](https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html?highlight=pxd#pxd-files) to declare the c++ API.
`pxd` files are organized in `_lib/cpp` and should maintain the same folder structure as c++ API hierarchy.

<!-- Add reference when ready -->
<!-- `pyx` files should be organized in similar categories as `library_design` -->

In general,
cuSpatial inputs should be device copiable and array-like.
`cudf.Series` constructor converts various array like inputs into device arrays.
Due to historical reasons,
`cudf.column.column.as_column` is widely used throughout the code base,
it is discouraged since recent cuDF refactoring defined that column APIs as internal APIs.
Most libcuspatial APIs has input type constraints,
and the python API should perform type checks and casts to meet its requirement.
`column_utils` provides a some useful utilities to normalize input columns.
