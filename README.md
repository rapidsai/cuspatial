# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;cuSpatial - GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cuspatial/job/branches/job/cuspatial-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/cuspatial/job/branches/job/cuspatial-branch-pipeline/)

**NOTE:** cuSpatial depends on [cuDF](https://github.com/rapidsai/cudf) and
[RMM](https://github.com/rapidsai/rmm) from [RAPIDS](https://rapids.ai/).

## Operations

cuSpatial supports the following operations on spatial and trajectory data:

1. Spatial window query
2. Point-in-polygon test
3. Haversine distance
4. Hausdorff distance
5. Deriving trajectories from point location data
6. Computing distance/speed of trajectories
7. Computing spatial bounding boxes of trajectories
8. Quadtree-based indexing for large-scale point data
9. Quadtree-based point-in-polygon spatial join
10. Quadtree-based point-to-linestring nearest neighbor distance
11. Distance computation (point-point, point-linestring, linestring-linestring)
12. Finding nearest points between point and linestring

Future support is planned for the following operations:

1. Temporal window query
2. Temporal point query (year+month+day+hour+minute+second+millisecond)
3. Grid-based indexing for points and polygons
4. R-Tree-based indexing for Polygons/Polylines

## Install from Conda

To install via conda:

```shell
conda install -c conda-forge -c rapidsai-nightly cuspatial
```

## Install from Source

To build and install cuSpatial from source:

### Pre-requisite

- gcc >= 7.5
- cmake >= 3.23
- miniconda

### Fetch cuSpatial repository

```shell
export `CUSPATIAL_HOME=$(pwd)/cuspatial` && \
git clone https://github.com/rapidsai/cuspatial.git $CUSPATIAL_HOME
```
### Install dependencies

1. `export CUSPATIAL_HOME=$(pwd)/cuspatial`
2. clone the cuSpatial repo

```shell
conda env update --file conda/environments/all_cuda-115_arch-x86_64.yaml 
```

### Build and install cuSpatial

1. Compile and install
   ```shell
   cd $CUSPATIAL_HOME && \
   chmod +x ./build.sh && \
   ./build.sh
   ```

2. Run C++/Python test code

   Some tests using inline data can be run directly, e.g.:

   ```shell
   $CUSPATIAL_HOME/cpp/build/gtests/LEGACY_HAUSDORFF_TEST
   $CUSPATIAL_HOME/cpp/build/gtests/POINT_IN_POLYGON_TEST
   python python/cuspatial/cuspatial/tests/legacy/test_hausdorff_distance.py
   python python/cuspatial/cuspatial/tests/test_pip.py
   ```

   Some other tests involve I/O from data files under `$CUSPATIAL_HOME/test_fixtures`.
   For example, `$CUSPATIAL_HOME/cpp/build/gtests/SHAPEFILE_READER_TEST` requires three
   pre-generated polygon shapefiles that contain 0, 1 and 2 polygons, respectively. They are available at
   `$CUSPATIAL_HOME/test_fixtures/shapefiles` <br>

**NOTE:** Currently, cuSpatial supports reading point/polyine/polygon data using
Structure of Array (SoA) format and a [shapefile reader](./cpp/src/io/shp)
to read polygon data from a shapefile.
Alternatively, python users can read any point/polyine/polygon data using
existing python packages, e.g., [Shapely](https://pypi.org/project/Shapely/)
and [Fiona](https://github.com/Toblerity/Fiona),to generate numpy arrays and feed them to
[cuSpatial python APIs](https://docs.rapids.ai/api/cuspatial/stable/).
