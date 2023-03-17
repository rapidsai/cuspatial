# Build and Install cuSpatial From Source

## Pre-requisites

- gcc >= 7.5
- cmake >= 3.23
- miniconda

## Fetch cuSpatial repository

```shell
export `CUSPATIAL_HOME=$(pwd)/cuspatial` && \
git clone https://github.com/rapidsai/cuspatial.git $CUSPATIAL_HOME
```
## Install dependencies

1. `export CUSPATIAL_HOME=$(pwd)/cuspatial`
2. clone the cuSpatial repo

```shell
conda env update --file conda/environments/all_cuda-118_arch-x86_64.yaml
```

## Build and install cuSpatial

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
