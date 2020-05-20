# cuSpatial 0.14.0 (Date TBD)

## New Features
- PR #143 Support constructing quadtrees on point data
- PR #182 Local gpuCI build script
- PR #208 NYC Taxi Years Correlation Notebook (thanks @taureandyernv)

## Improvements

- PR #147 Update Conda/CMake configs to match other RAPIDS projects
- PR #163 Fix cudf legacy Cython imports/cimports
- PR #166 Move trajectory.hpp files to legacy
- PR #167 Align utility.hpp with libcudf style
- PR #171 Update trajectory.hpp to libcudf++
- PR #173 Move hausdorff.hpp files to legacy
- PR #172 Move coordinate_transform.hpp files to legacy
- PR #170 Update coordinate_transform.hpp to libcudf++
- PR #174 Update hausdorff.hpp to libcudf++
- PR #183 Add libcuspatial benchmark scaffolding
- PR #186 Move haversine.hpp files to legacy
- PR #194 Add .clang-format & format all files
- PR #190 Port coordinate_transform.hpp cython files
- PR #191 Move point_in_polygon.hpp files to legacy
- PR #193 Move shapefile_readers.hpp files to legacy
- PR #196 Move utilities/utilities.hpp to legacy
- PR #195 Fix PIP docs
- PR #197 Move query.hpp files to legacy
- PR #198 Port spatial_window queries to libcudf++
- PR #192 Update point_in_polygon.hpp to libcudf++
- PR #201 Update trajectory cython to libcudf++
- PR #189 Update haversine.hpp files to libcudf++
- PR #200 Update shapefile_readers.hpp to libcudf++
- PR #203 Port point_in_polygon.hpp cython files
- PR #202 Update haversine cython to libcudf++
- PR #204 Port shapefile_readers.hpp cython files
- PR #205 Port hausdorff.hpp cython to libcudf++
- PR #206 Remove legacy code.

## Bug Fixes

- PR #141 Fix dangling exec_policy pointer and invalid num_ring argument.
- PR #169 Fix shapefile reader compilation with GCC 7.x / CUDA 10.2
- PR #178 Fix broken haversine tests introduced by upstream CUDF PRs.
- PR #175 Address RMM API changes by eliminating the use of the RMM_API
- PR #199 Fix coordinate transform tests


# cuSpatial 0.13.0 (31 Mar 2020)

## New Features

- PR #126 Create and build cuSpatial Docs 
- PR #130 Add cubic spline fit and interpolation

## Improvements

- PR #128 Use RMM's `DeviceBuffer` for Python allocations
- PR #142 Disable deprecation warnings by default
- PR #138 Update Build instructions in the README

## Bug Fixes

- PR #123 Update references to error utils after libcudf changes
- PR #136 Remove build erroring for deprecation warnings


# cuSpatial 0.12.0 (04 Feb 2020)

## New Features

## Improvements

- PR #109 Update OPS codeowners group name
- PR #113 Support libcudf++

## Bug Fixes

- PR #116 Fix API issue with shapefile reader


# cuSpatial 0.11.0 (11 Dec 2019)

## New Features

- PR #86 Add Shapefile reader for polygons
- PR #92 Python bindings for shapefile reader

## Improvements

- PR #104 Remove unused CUDA conda labels

## Bug Fixes

- PR #94 Add legacy headers as cudf migrates
- PR #98 Updates to accommodate cudf refactoring
- PR #103 Update the include paths for cuda_utils


# cuSpatial 0.10.0 (16 Oct 2019)

## New Features

- PR #7 Initial code
- PR #18 Python initial unit tests and bindings
- PR #32 Python API first pass
- PR #37 Python __init__.py package design
- PR #38 Add __init__.py empties to resolve issue with PYTHONPATH
- PR #25 Add gpuCI integration

## Improvements

- PR #31 Add Github CODEOWNERS
- PR #39 Add cython headers to install, python / cmake packaging cleanup
- PR #41 Python and Cython style cleanup, pre-commit hook
- PR #44 Update all demos with Python API
- PR #45 Improve documentation in haversine and point in polygon
- PR #50 Validate that distance and speed work with all datetimes
- PR #58 Hausdorff distance returns a DataFrame, and better docs.
- PR #61 Point-in-polygon DataFrame output
- PR #59 Improve detail of point in polygon docs
- PR #64 Use YYMMDD tag in nightly build
- PR #68 Use YYMMDD tag in nightly build of cuspatial python
- PR #97 Drop `cython` from run requirements
- PR #82 Added update-version.sh
- PR #86 Add Shapefile reader for polygons

## Bug Fixes

- PR #16 `cuspatial::subset_trajectory_id()` test improvements and bug fixes
- PR #17 Update issue / PR templates
- PR #23 Fix cudf Cython imports
- PR #24 `cuspatial::derive_trajectories()` test improvements and bug fixes
- PR #33 `cuspatial::trajectory_distance_and_speed()` test improvements and bug fixes
- PR #49 Docstring for haversine and argument ordering was backwards
- PR #66 added missing header in tests
- PR #70 Require width parameterization of bitmap to binary conversion
