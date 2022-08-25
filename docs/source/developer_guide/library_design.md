# cuSpatial Library Design

cuSpatial has two main components: the cuSpatial Python package and the `libcuspatial` C++ library,
referred to as `cuspatial` and `libcuspatial` respectively in this documentation. This page
discusses the design of `cuspatial`. For information on `libcuspatial`, see the [libcuspatial
developer guide](TODO link) and [C++ API reference](TODO link).

## Overview

At a high level, `cuspatial` has three parts:
- A GPU backed `GeoDataFrame` data structure
- A set of computation APIs
- A Cython API layer

## GPU Accelerated `GeoDataFrame` and `GeoSeries`

Note: the core data structure of cuSpatial shares the same name as that of `geopandas`, so we refer
to geopandas' dataframe object as `geopandas.GeoDataFrame` and to cuspatial's dataframe object as
`GeoDataFrame`.

Under the hood,
cuspatial can perform parallel computation on geometry data thanks to its [structure of array](https://en.wikipedia.org/wiki/Parallel_array) (SoA) format.
Specifically,
cuspatial adopts geoarrow format as the SoA standard.
Geoarrow is a derived data type from the arrow list type adopting a [`Variable-size List Layout`](https://arrow.apache.org/docs/format/Columnar.html#variable-size-list-layout),
with the inner-most layer storing the points with a `Fixed-size list layout` array with `size=2`.
Per definition,
each increase in geometry complexity (dimension, or multi-geometry) requires an extra level of indirection.
Geoarrow allows a mixture of geometry types to present in the same column by adopting the [Dense Union Array Layout](https://arrow.apache.org/docs/format/Columnar.html#dense-union).
In cusptial,
we refer to each level of indirection from highest level to lowest by
`geometries`, `parts`, `rings` and `coordinates`.
The first three are integral offset arrays and the last is an floating-point interleaved xy-coordinated array.
Read [geoarrow format documentation](https://github.com/geopandas/geo-arrow-spec/blob/main/format.md) specification for more detail.

Cuspatial implements partial arrow union array via `GeoColumn` and `GeoMeta`.
A GeoColumn is a composition of child columns and a `GeoMeta` object.
The `GeoMeta` owns two arrays that are similar to the types buffer and offsets buffer from dense union.

```{note}
Currently, `GeoColumn` only implements four concrete array types: `points`,
`multipoints`, multilinestrings (called `lines`) and multipolygons (called
`polygons`). Linestrings and multilinestrings are stored uniformly as
multilinestrings in the `multilinestrings` array. Polygons and multipolygons are
stored uniformly as multipolygons in the `multipolygons` array.

Points and multipoints are stored separately in different arrays, because
storing points in a multipoints array requires 50% more storage overhead.
While this may also be true for linestrings and polygons, many uses of
cuSpatial involve more complex linestrings and polygons, where the 
storage overhead of multigeometry indirection is lower compared to points.

`GeoSeries` and `GeoDataFrame` inherit from `cudf.Series` and
`cudf.DataFrame` respectively. `Series` and `DataFrame` are both generic
`Frame` objects which represent a collection of generic columns. cuSpatial
extends these cuDF objects by allowing `GeoColumn`s to be present in the
frame.

`GeoSeries` and `GeoDataFrame` are convertible to and from `geopandas`.
Interoperability between cuspatial, `geopandas` and other data formats is
maintained in the `cuspatial.io` package.

### UnionArray Compliance

As previously mentioned,
cuspatial's `GeoColumn` is an specialization of arrow's `UnionArray`.
A fundamental addition in data types should be implemented in `cudf` and `GeoColumn` should simply inherits its functionality.
However,
`UnionArray` stands distinctly from existing data types in libcudf and requires substantial effort to implement.
In the interim,
cuspatial developers should build `GeoColumn` complying to `UnionArray` standards.
Such effort may be upstreamed to `cudf` when development is more underway.

## GIS computation APIs

Besides the data structure,
cuspatial maintains a set of computation APIs.
The computation APIs are organized into several modules.
All spatial computation modules are further grouped into a `spatial` subpackage.
Developers are encouraged to use specific names for the module of the function added in the PR.

### Legacy and Modern APIs

For historical reasons,
cuspatial APIs exposes raw array inputs for users to provide raw geometry coordinate array and offsets.
Newer APIs should accept a `GeoSeries` or `GeoDataFrame` as input.
Developers may extract geometry offsets and coordinates via geometry accessors provided,
such as `GeoSeries.point`, `GeoSeries.multipoint`, `GeoSeries.lines`, `GeoSeries.polygon`.

## Cython Layer

The lowest layer of cuspatial is its interaction with `libcuspatial` via Cython.
The Cython layer is composed of two components: C++ bindings and
Cython wrappers. The first component consists of
[`.pxd` files](https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html),
which are Cython declaration files that expose the contents of C++ header
files to other Cython files. The second component consists of Cython
wrappers for this functionality. These wrappers are necessary to expose
this functionality to pure Python code.

To interact with the column-based APIs in `libcuspatial`, developers should
have basic familiarity with `libcudf` objects. `libcudf` is built around two
principal objects whose names are largely self-explanatory: `column` and
`table`. `libcudf` also defines corresponding non-owning "view" types
`column_view` and `table_view`. Both `libcudf` and `libcuspatial` APIs
typically accept views and return owning types. When a `cuspatial` object
owns one ore more c++ owning objects, the lifetime of these objects is
automatically managed by python's reference counting mechanism.

Similar to cuDF, Cython wrappers must convert `Column` objects into
`column_view` objects, call the `libcuspatial` API, and reconstruct a cuDF
object from the c++ result. By the time code reaches this stage, the
objects are assumed to be fully legal inputs to the `libcuspatial` API.
Therefore the wrapper should not contain additional components besides
the above.
