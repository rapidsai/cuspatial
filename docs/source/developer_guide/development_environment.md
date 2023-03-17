# Creating a Development Environment

cuSpatial follows the RAPIDS release schedule, so developers are encouraged to develop 
using the latest development branch of RAPIDS libraries that cuspatial depends on. Other
cuspatial dependencies can be found in `conda/environments/`.

Maintaining a local development environment can be an arduous task, especially after each
RAPIDS release. Most cuspatial developers today use
[rapids-compose](https://github.com/trxcllnt/rapids-compose) to setup their development environment.
It contains helpful scripts to build a RAPIDS development container image with the required
dependencies and RAPIDS libraries automatically fetched and correctly versioned. It also provides
script commands for simple building and testing of all RAPIDS libraries, including cuSpatial.
`rapids-compose` is the recommended way to set up your environment to develop for cuspatial.

### To build and install cuSpatial from source:

#### Pre-requisite

- gcc >= 7.5
- cmake >= 3.23
- miniconda

#### Fetch cuSpatial repository

```shell
export `CUSPATIAL_HOME=$(pwd)/cuspatial` && \
git clone https://github.com/rapidsai/cuspatial.git $CUSPATIAL_HOME
```
#### Install dependencies

1. `export CUSPATIAL_HOME=$(pwd)/cuspatial`
2. clone the cuSpatial repo

```shell
conda env update --file conda/environments/all_cuda-115_arch-x86_64.yaml
```

#### Build and install cuSpatial

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
