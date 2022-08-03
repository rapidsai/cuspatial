# cuSpatial Library Design

At a high level, cuSpatial comprises of two components.
The GIS computation APIs and the GPU accelerated Geoarrow data structure.

## GIS computation APIs

In general, computation APIs accepts structure of arrays inputs.
The inputs should be convertible to device arrays via `as_column` call in cudf.
Most libcuspatial APIs has input type constraints,
and the python API should perform type checks and casts to meet its requirement.
`column_utils` provides a some useful utilities to normalize input columns.

<!-- The below require certain refactor to take place before we can add this. -->
<!-- The GIS computation APIs supports various accelerated GIS algorithms.
cuSpatial categorizes the computation APIs into four types:
- Spatial Functions
- Indexing
- Join
- Trajectory Computation
 -->

## GPU Accelerated Geoarrow Data Structure

TBA by @thomcom, can be ported from internal.md
