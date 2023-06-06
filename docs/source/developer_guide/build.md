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

### From devcontainer:

Execute `build-cuspatial-cpp -DBUILD_TESTS=ON` to build libcuspatial and tests.
In addition, `build-cuspatial-python` to build cuspatial cython components.

### From bare metal:

Compile libcuspatial (C++), cuspatial (cython) and C++ tests:
```shell
cd $CUSPATIAL_HOME && \
chmod +x ./build.sh && \
./build.sh libcuspatial cuspatial tests
```

## Validate installation by running C++ and Python tests

C++ tests locate under `$CUSPATIAL_HOME/cpp/build/gtests`. Python tests locate under
`$CUSPATIAL_HOME/python/cuspatial/cuspatial/tests`.

```note
Devcontainer users: to manage difference between branches, the build directory is further alternatively placed
`$CUSPATIAL_HOME/cpp/build/release` and `$CUSPATIAL_HOME/cpp/build/latest`, where `release` places the release build,
and `latest` is the symlink to the most recent build directory.
```

Execute C++ tests:
```shell
$CUSPATIAL_HOME/cpp/build/gtests/
$CUSPATIAL_HOME/cpp/build/gtests/POINT_IN_POLYGON_TEST
```

Execute Python tests:
```
python python/cuspatial/cuspatial/tests/test_hausdorff_distance.py
python python/cuspatial/cuspatial/tests/test_pip.py
```
