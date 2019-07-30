# cuSpatial
GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library <br>
**NOTE:** cuSpatial depends on [cudf](https://github.com/rapidsai/cudf) and [rmm](https://github.com/rapidsai/rmm) under [RAPDIS](https://rapids.ai/) framework<br> 
See [here](https://nvidia-my.sharepoint.com/:p:/r/personal/jiantingz_nvidia_com/Documents/GPU4STA_V5.pptx?d=wa5b5d6d397074ea9a1600e74fd8a6345&csf=1&e=h7MdRq) 
for a brief summary/discussion on C++ backend performance (as standalone components and with comparions to serial/mutli-core implementations on CPUs and/or legacy code) <br>
See the [deisgn documentation](doc/design.md) for a breif description on how spatial and trajectory data are represented in cuSpatial and the graph of operations on them.   

<h2>Implemented operations:</h2> 
Currently, cuSpatial supports a subset of operations for spatial and trajectory data: <br>
1) [spatial window query](cpp/src/stq) <br>
2) [point-in-polygon test](cpp/src/spatial) <br>
3) [deriving trajectories from point location data](cpp/src/traj) <br>
4) [computing distance/speed of trajectories](cpp/src/traj) <br>
5) [computing spatial bounding boxes of trajectories](cpp/src/traj) <br> 
<br>
Another subset of operations will be added shortly: <br>
1) temporal window query (cpp/src/stq) <br>
2) temporal point query (year+month+day+hour+minute+second+millisecond)(cpp/src/stq)<br>
3) quadtree-based indexing for large-scale point data (cpp/src/idx)<br>
4) point-to-polyline nearest distance/neighbor (cpp/src/spatial)<br>
<br>
More operations are being planned/developed. 
 
<h2>Compile/Install C++ backend</h2>
To compile and run cuSpatial, use the following steps <br>
CUSPATIAL_HOME=$(pwd)/cuspatial <br>
Step 1: clone a copy of cuSpatial (using your nvidia git-lab username/password) <br>
git clone https://gitlab-master.nvidia.com/jiantingz/cuspatial ${CUSPATIAL_HOME}<br>
<br>

Step 2: install cudf by following the [instructions](https://github.com/rapidsai/cudf/blob/branch-0.9/CONTRIBUTING.md) <br>
The rest of steps assume CUDACXX=/usr/local/cuda/bin/nvcc, CUDF_HOME=$(pwd)/cudf are set and conda environment cudf_dev is activated after Step 2. <br>

Step 3: copy [cub](https://github.com/NVlabs/cub) and [dlpack](https://github.com/rapidsai/dlpack/) from cudf to cuSpatial<br>

mkdir $CUSPATIAL_HOME/cpp/thirdparty/ <br>
cp -r $CUDF_HOME/cpp/thirdparty/cub     $CUSPATIAL_HOME/cpp/thirdparty/<br> 
cp -r $CUDF_HOME/cpp/thirdparty/dlpack    $CUSPATIAL_HOME/cpp/thirdparty/<br> 

Step 4: comile and install C++ backend <br>

cd $CUSPATIAL_HOME/cpp <br>
mkdir build <br>
cd build <br>
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX <br>
make <br>
make install <br>

cuSpatial should be installed at $CONDA_PREFIX, e.g., /home/jianting/anaconda3/envs/cudf_dev <br>
For cuspatial, the include path is $CONDA_PREFIX/include/cuspatial/ and the library path  $CONDA_PREFIX/lib/libcuspatial.so, respetively. 

<h2>Compile/Install Python wrapper and run Python test code </h2> 

Step 6: build and install python wrapper <br>

First, make a copy of cudf header files and put it under cuSpatial include directory to make setup easier <br> 

cp -r $CUDF_HOME/cpp/include/cudf $CUSPATIAL_HOME/cpp/include <br>

Then, get started:<br> 

cd $CUSPATIAL_HOME/python/cuspatial <br>
python setup.py build_ext --inplace <br>
python setup.py install <br>

Step 7: Run python test code <br>

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
to generate numpy arrays and feed them to [cuSpatial python APIs](python/cudf/cudf/bindings). <br> 




