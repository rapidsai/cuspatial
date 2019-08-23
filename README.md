# cuSpatial
## GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library
**NOTE:** cuSpatial depends on [cuDF](https://github.com/rapidsai/cudf) and [RMM](https://github.com/rapidsai/rmm) under [RAPIDS](https://rapids.ai/) framework<br> 
See [here](https://nvidia-my.sharepoint.com/:p:/r/personal/jiantingz_nvidia_com/Documents/GPU4STA_V5.pptx?d=wa5b5d6d397074ea9a1600e74fd8a6345&csf=1&e=h7MdRq) 
for a brief summary/discussion on C++ backend performance (as standalone components and with comparions to serial/multi-core implementations on CPUs and/or legacy code) <br>
See the [design documentation](doc/design.md) for a brief description of how spatial and trajectory data are represented in cuSpatial and the graph of operations on them.   

## Implemented operations:
cuSpatial supports the following operations on spatial and trajectory data:
1. [Spatial window query](cpp/src/stq)
2) [Point-in-polygon test](cpp/src/spatial) <br>
3) [Harversine distance](cpp/src/spatial) <br>
4) [Hausdorff distance](cpp/src/spatial)<br>
5) [Deriving trajectories from point location data](cpp/src/traj) <br>
6) [Computing distance/speed of trajectories](cpp/src/traj) <br>
7) [Computing spatial bounding boxes of trajectories](cpp/src/traj) <br> 

Future support is planned for the following operations.
1. Temporal window query (cpp/src/stq)
2) Temporal point query (year+month+day+hour+minute+second+millisecond)(cpp/src/stq)<br>
3) Point-to-polyline nearest neighbor distance](cpp/src/spatial) <br>
4) Grid based indexing for points and polygons (cpp/src/idx)<br>
5) Quadtree based indexing for large-scale point data (cpp/src/idx)<br>
6) R-Tree based indexing for Polygons/Polylines (cpp/src/idx)<br>
 
## Compile/Install C++ backend
To compile and run cuSpatial, use the following steps <br>
export CUSPATIAL_HOME=$(pwd)/cuspatial <br>
Step 1: clone a copy of cuSpatial (using your nvidia git-lab username/password) <br>
git clone https://github.com/zhangjianting/cuspatial.git -b fea-initial-code ${CUSPATIAL_HOME}<br>
<br>

Step 2: install cudf by following the [instructions](https://github.com/rapidsai/cudf/blob/branch-0.9/CONTRIBUTING.md) <br>
As a shortcut, one can just run the scripts "./build.sh libcudf" and "./build.sh cudf" under {CUSPATIAL_HOME} <br>
The rest of steps assume "export CUDACXX=/usr/local/cuda/bin/nvcc" and "export CUDF_HOME=$(pwd)/cudf are executed, and conda environment cudf_dev is activated after Step 2. <br>
Please note that, on some environments (OS+BASH), CUDF_HOME set by using "CUDF_HOME=..." when installing cuDF can not be used in cuSpatial. 
Please use <b>export CUDF_HOME=$(pwd)/cudf</b> instead.   

Step 3: compile and install C++ backend <br>

cd $CUSPATIAL_HOME/cpp <br>
mkdir build <br>
cd build <br>
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX <br>
make (or make -j [n]) <br>
make install <br>

cuSpatial should be installed at $CONDA_PREFIX, e.g., /home/jianting/anaconda3/envs/cudf_dev <br>
For cuspatial, the include path is $CONDA_PREFIX/include/cuspatial/ and the library path  $CONDA_PREFIX/lib/libcuspatial.so, respetively. 

<h2>Compile/Install Python wrapper and run Python test code </h2> 

Step 4: build and install python wrapper <br>

cd $CUSPATIAL_HOME/python/cuspatial <br>
python setup.py build_ext --inplace <br>
python setup.py install <br>

Step 5: Run python test code <br>

First,cuSpatial Python API path to PYTHONPATH (there are tools under tests subdir), i.e., <br>
export PYTHONPATH=$CUSPATIAL_HOME/python/cuspatial <br>

Some test code using toy data can be run directly, e.g., <br>
python  $CUSPATIAL_HOME/python/cuspatial/cuspatial/tests/pip2_test_soa_toy.py <br>

However, many test code uses real data from an ITS (Intelligent Transportation System) application. 
You will need to follow instructions at [data/README.md](./data/README.md) to generate data for these test code. <br>
Alternatively, you can download the preprocessed data("locust.*", "its_4326_roi.*", "itsroi.ply" and "its_camera_2.csv") from [here](https://nvidia-my.sharepoint.com/:u:/p/jiantingz/EdHR7qlaRSVPtw46XYVR9sQBjCcnUHygCuPUC3Hf8gW73A?e=LCr9nK),
extrat the files and put them directly under $CUSPATIAL_HOME/data for quick demos. <br>
A brief discription of these data files and their semantic roles in the ITS application can be found [here](doc/itsdata.md) 

After data are dowloaded and/or pre-processed, you can run the [python test code](python/cuspatial/cuspatial/tests), e.g., <br>
python  $CUSPATIAL_HOME/python/cuspatial/cuspatial/tests/pip2_verify.py <br>
python  $CUSPATIAL_HOME/python/cuspatial/cuspatial/tests/traj2_test_soa3.py <br>
python  $CUSPATIAL_HOME/python/cuspatial/cuspatial/tests/stq_test_soa1.py <br>

<br>
**NOTE:** Currently, cuSpatial supports reading point/polyine/polygon data using Structure of Array (SoA) format (more readers are being developed) <br>
Alternatively, python users can read any point/polyine/polygon data using existing python packages, e.g., [Shapely](https://pypi.org/project/Shapely/), 
to generate numpy arrays and feed them to [cuSpatial python APIs](python/cuspatial/cuspatial/bindings). <br> 




