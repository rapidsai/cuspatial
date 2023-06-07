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
conda env create -n cuspatial --file conda/environments/all_cuda-118_arch-x86_64.yaml
```

## Build cuSpatial

### From the cuSpatial Dev Container:

Execute `build-cuspatial-cpp to build `libcuspatial`. The following options may be added.
 - `-DBUILD_TESTS=ON`: build `libcuspatial` unit tests. 
 - `-DBUILD_BENCHMARKS=ON`: build `libcuspatial` benchmarks.  
 - `-DCMAKE_BUILD_TYPE=Debug`: Create a Debug build of `libcuspatial` (default is Release).
In addition, `build-cuspatial-python` to build cuspatial cython components.

### From Bare Metal:

Compile libcuspatial (C++), cuspatial (cython) and C++ tests:
```shell
cd $CUSPATIAL_HOME && \
chmod +x ./build.sh && \
./build.sh libcuspatial cuspatial tests
```

## Validate Installation with C++ and Python Tests

- C++ tests are located within the `$CUSPATIAL_HOME/cpp/build/gtests` directory.
- Python tests are located within the `$CUSPATIAL_HOME/python/cuspatial/cuspatial/tests` directory.

```note
Dev Container users:
To manage difference between branches and build types, the build directories are located at
`$CUSPATIAL_HOME/cpp/build/[release|debug]` depending on build type, and  `$CUSPATIAL_HOME/cpp/build/latest`.
is a symbolic link to the most recent build directory.

Execute C++ tests:
```shell
$CUSPATIAL_HOME/cpp/build/gtests/HAUSDORFF_TEST
$CUSPATIAL_HOME/cpp/build/gtests/POINT_IN_POLYGON_TEST_EXP
```

Execute Python tests:
```
python python/cuspatial/cuspatial/tests/test_geoseries.py
python python/cuspatial/cuspatial/tests/test_trajectory.py
```
