# cuSpatial 0.10.0 (Date TBD)

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
- PR #61 Point-in-polygon DataFrame output
- PR #64 Use YYMMDD tag in nightly build

## Bug Fixes

- PR #16 `cuspatial::subset_trajectory_id()` test improvements and bug fixes
- PR #17 Update issue / PR templates
- PR #23 Fix cudf Cython imports
- PR #24 `cuspatial::derive_trajectories()` test improvements and bug fixes
- PR #33 `cuspatial::trajectory_distance_and_speed()` test improvements and bug fixes
- PR #49 Docstring for haversine and argument ordering was backwards
- PR #66 added missing header in tests
