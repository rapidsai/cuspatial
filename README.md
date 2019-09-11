# cuSpatial
## GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library

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
To insall via conda, please run:
`conda install -c rapidsai-nightly cuspatial`

## Install from Source
To compile and install cuSpatial's C++ backend, please follow the following steps:

### Install dependencies

Currently, building cuSpatial requires a source installation of cuDF. Install
cuDF by following the [instructions](https://github.com/rapidsai/cudf/blob/branch-0.10/CONTRIBUTING.md#script-to-build-cudf-from-source)

The rest of steps assume the environment variable `CUDF_HOME` points to the 
root directory of your clone of the cuDF repo, and that the `cudf_dev` Anaconda
environment created in step 3 is active.

### Clone, build and install cuSpatial

1. export `CUSPATIAL_HOME=$(pwd)/cuspatial`
2. clone the cuSpatial repo

```
git clone https://github.com/rapidsai/cuspatial.git $CUSPATIAL_HOME
```

3. Compile and install C++ backend

```
cd $CUSPATIAL_HOME/cpp
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make # (or make -j [n])
make install
```

cuSpatial should now be installed at `$CONDA_PREFIX`. The cuspatial include path
is `$CONDA_PREFIX/include/cuspatial/` and the library path is
`$CONDA_PREFIX/lib/libcuspatial.so`. 

4. Compile and install cuSpatial Python wrapper and run Python test code

```
cd $CUSPATIAL_HOME/python/cuspatial
python setup.py build_ext --inplace
python setup.py install
```

5. Run python test code <br>

First, add the cuSpatial Python API path to `PYTHONPATH` (there are tools under
tests subdir): `export PYTHONPATH=$CUSPATIAL_HOME/python/cuspatial`

Some tests using toy data can be run directly, e.g.,

```
python  $CUSPATIAL_HOME/python/cuspatial/cuspatial/tests/pip2_test_soa_toy.py
```

However, many test code uses real data from an ITS (Intelligent Transportation
System) application. You will need to follow instructions at
[data/README.md](./data/README.md) to generate data for these test code.
Alternatively, you can download the preprocessed data ("locust.*",
"its_4326_roi.*", "itsroi.ply" and "its_camera_2.csv") from 
[here](https://nvidia-my.sharepoint.com/:u:/p/jiantingz/EdHR7qlaRSVPtw46XYVR9sQBjCcnUHygCuPUC3Hf8gW73A?e=LCr9nK).
Extract the files and put them directly under $CUSPATIAL_HOME/data for quick
demos. A brief description of these data files and their semantic roles in the
ITS application can be found [here](doc/itsdata.md) TODO THIS IS MISSING

After data are downloaded and/or pre-processed, you can run the 
[python test code](python/cuspatial/cuspatial/tests):

```
python  $CUSPATIAL_HOME/python/cuspatial/cuspatial/tests/pip2_verify.py
python  $CUSPATIAL_HOME/python/cuspatial/cuspatial/tests/traj2_test_soa3.py
python  $CUSPATIAL_HOME/python/cuspatial/cuspatial/tests/stq_test_soa1.py
```

**NOTE:** Currently, cuSpatial supports reading point/polyine/polygon data using
Structure of Array (SoA) format (more readers are being developed).
Alternatively, python users can read any point/polyine/polygon data using
existing python packages, e.g., [Shapely](https://pypi.org/project/Shapely/),
to generate numpy arrays and feed them to
[cuSpatial python APIs](python/cuspatial/cuspatial).
