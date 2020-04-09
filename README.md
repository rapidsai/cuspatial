# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuSpatial - GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cuspatial/job/branches/job/cuspatial-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cuspatial/job/branches/job/cuspatial-branch-pipeline/)

**NOTE:** cuSpatial depends on [cuDF](https://github.com/rapidsai/cudf) and
[RMM](https://github.com/rapidsai/rmm) from [RAPIDS](https://rapids.ai/).

## Implemented operations:
cuSpatial supports the following operations on spatial and trajectory data:
1. Spatial window query
2. Point-in-polygon test
3. Haversine distance
4. Hausdorff distance
5. Deriving trajectories from point location data
6. Computing distance/speed of trajectories
7. Computing spatial bounding boxes of trajectories

Future support is planned for the following operations.
1. Temporal window query
2. Temporal point query (year+month+day+hour+minute+second+millisecond)
3. Point-to-polyline nearest neighbor distance
4. Grid-based indexing for points and polygons
5. Quadtree-based indexing for large-scale point data
6. R-Tree-based indexing for Polygons/Polylines

## Install from Conda
To install via conda:
```
conda install -c conda-forge -c rapidsai-nightly cuspatial
```

## Install from Source
To build and install cuSpatial from source:

### Install dependencies

Currently, building cuSpatial requires a source installation of cuDF. Install
cuDF by following the [instructions](https://github.com/rapidsai/cudf/blob/branch-0.11/CONTRIBUTING.md#script-to-build-cudf-from-source)

The rest of steps assume the environment variable `CUDF_HOME` points to the 
root directory of your clone of the cuDF repo, and that the `cudf_dev` Anaconda
environment created in step 3 is active.

### Clone, build and install cuSpatial

1. export `CUSPATIAL_HOME=$(pwd)/cuspatial`
2. clone the cuSpatial repo

```
git clone --recurse-submodules https://github.com/rapidsai/cuspatial.git $CUSPATIAL_HOME
```

3. Compile and install 
Similar to cuDF (version 0.11), simplely run 'build.sh' diectly under $CUSPATIAL_HOME<br>
Note that a "build" dir is created automatically under $CUSPATIAL_HOME/cpp

4. Run C++/Python test code <br>

Some tests using inline data can be run directly, e.g.,
```
$CUSPATIAL_HOME/cpp/build/gtests/LEGACY_HAUSDORFF_TEST
$CUSPATIAL_HOME/cpp/build/gtests/POINT_IN_POLYGON_TEST
python python/cuspatial/cuspatial/tests/legacy/test_hausdorff_distance.py
python python/cuspatial/cuspatial/tests/test_pip.py
```

Some other tests involve I/O from data files under $CUSPATIAL_HOME/test_fixtures.
For example, $CUSPATIAL_HOME/cpp/build/gtests/SHAPEFILE_POLYGON_READER_TEST requires three
pre-generated polygon shapefiles that contain 0, 1 and 2 polygons, respectively. They are available at 
$CUSPATIAL_HOME/test_fixtures/shapefiles <br>

**NOTE:** Currently, cuSpatial supports reading point/polyine/polygon data using
Structure of Array (SoA) format and a [shapefile reader](./cpp/src/io/shp)
to read polygon data from a shapefile.
Alternatively, python users can read any point/polyine/polygon data using
existing python packages, e.g., [Shapely](https://pypi.org/project/Shapely/) 
and [Fiona](https://github.com/Toblerity/Fiona),to generate numpy arrays and feed them to
[cuSpatial python APIs](python/cuspatial/cuspatial).
