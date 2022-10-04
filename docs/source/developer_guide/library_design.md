# cuSpatial Library Design

## Overview

At a high level, `cuspatial` has three parts:
- A GPU backed `GeoDataFrame` data structure
- A set of computation APIs
- A Cython API layer

## Core Data Structures 

```{note}
Note: the core data structure of cuSpatial shares the same name as that of `geopandas`, so we refer
to geopandas' dataframe object as `geopandas.GeoDataFrame` and to cuspatial's dataframe object as
`GeoDataFrame`.
```

### Introduction to GeoArrow Format

Under the hood, cuspatial can perform parallel computation on geometry
data thanks to its
[structure of arrays](https://en.wikipedia.org/wiki/Parallel_array) (SoA)
format. Specifically, cuspatial adopts GeoArrow format, which is an extension
to Apache Arrow format that uses Arrow's 
[`Variable-size List Layout`](https://arrow.apache.org/docs/format/Columnar.html#variable-size-list-layout)
to support geometry arrays.

By definition, each increase in geometry complexity (dimension, or multi-
geometry) requires an extra level of indirection. In cuSpatial, we use the following names for the levels of indirection from
highest level to lowest: `geometries`, `parts`, `rings` and `coordinates`. The
first three are integral offset arrays and the last is a floating-point
interleaved xy-coordinate array.

Geoarrow also allows a mixture
of geometry types to be present in the same column by adopting the
[Dense Union Array Layout](https://arrow.apache.org/docs/format/Columnar.html#dense-union).

Read the [geoarrow format specification](https://github.com/geopandas/geo-arrow-spec/blob/main/format.md)
for more detail.

### GeoColumn

cuSpatial implements a specialization of Arrow dense union via `GeoColumn` and
`GeoMeta`. A `GeoColumn` is a composition of child columns and a
`GeoMeta` object. The `GeoMeta` owns two arrays that are similar to the
types buffer and offsets buffer from Arrow dense union.

```{note}
Currently, `GeoColumn` implements four concrete array types: `points`,
`multipoints`, multilinestrings and multipolygons. Linestrings and
multilinestrings are stored uniformly as multilinestrings in the
`multilinestrings` array. Polygons and multipolygons are
stored uniformly as multipolygons in the `multipolygons` array.

Points and multipoints are stored separately in different arrays, because
storing points in a multipoints array requires 50% more storage overhead.
While this may also be true for linestrings and polygons, many uses of
cuSpatial involve more complex linestrings and polygons, where the 
storage overhead of multigeometry indirection is lower compared to points.
```

`GeoSeries` and `GeoDataFrame` inherit from `cudf.Series` and
`cudf.DataFrame` respectively. `Series` and `DataFrame` are both generic
`Frame` objects which represent a collection of generic columns. cuSpatial
extends these cuDF objects by allowing `GeoColumn`s to be present in the
frame.

`GeoSeries` and `GeoDataFrame` are convertible to and from `geopandas`.
Interoperability between cuspatial, `geopandas` and other data formats is
maintained in the `cuspatial.io` package.

### UnionArray Compliance

As previously mentioned, cuspatial's `GeoColumn` is a specialization of
Arrow's dense `UnionArray`. A fundamental addition to cuDF data types should be
implemented in cuDF so that `GeoColumn` can simply inherit its
functionality. However, dense `UnionArray` stands distinct from existing data types
in libcudf and requires substantial effort to implement. In the interim, 
cuSpatial provides a `GeoColumn` complying to the dense `UnionArray`
specification. This may be upstreamed to libcudf as it matures.

## Geospatial computation APIs

In addition to data structures, cuSpatial provides a set of computation APIs.
The computation APIs are organized into several modules. All spatial
computation modules are further grouped into a `spatial` subpackage.
Module names should correspond to a specific computation category,
such as `distance` or `join`. Cuspatial avoids using general category names,
such as `generic`.

### Legacy and Modern APIs

For historical reasons, older cuSpatial APIs expose raw array inputs for
users to provide raw geometry coordinate arrays and offsets. Newer Python
APIs should accept a `GeoSeries` or `GeoDataFrame` as input. Developers
may extract geometry offsets and coordinates via cuSpatial's geometry
accessors such as `GeoSeries.points`, `GeoSeries.multipoints`,
`GeoSeries.lines`, `GeoSeries.polygons`. Developer can then pass the geometries
offsets and corrdinate arrays to Cython APIs.

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
