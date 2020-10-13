# Java Bindings for Cuspatial

This project is a Java layer bridging Java user and libcuspatial.so 

Follow these instructions to create local builds and installation of librmm, libcudf and cudf-java, and libcuspatial

## Build and Install Dependencies:
### 1. libcudf and cudf-java
1.1) Follow instructions on this page for libcudf:
`https://github.com/rapidsai/cudf/blob/branch-0.11/CONTRIBUTING.md#script-to-build-cudf-from-source`,
but append two more cmake config flags `-DARROW_STATIC_LIB=ON -DBoost_USE_STATIC_LIBS=ON`,
changing
```xml 
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON
```
to 
```xml
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON -DARROW_STATIC_LIB=ON -DBoost_USE_STATIC_LIBS=ON
```
Make sure to `make install` after `make`. DO NOT use the `build.sh` script to build `libcudf.so`

1.2) Follow the instructions in `${CUDF_HOME}/java/README.md` to build and install cuDF product JAR and test JAR
```xml 
$ cd ${CUDF_HOME}/java 
$ mvn clean install
```
NOTE: 
1. Following above steps you will have conda env `cudf_dev` which is required for subsequent builds. 
2. libcudf.so will be installed under conda env `cudf_dev`.
3. libcudfJni.so will be built by maven pom.
4. cuDF JARs will be installed to maven local.

### 2. librmm (header-only)

`export RMM_HOME=${pwd}/rmm` and `git clone` source into `$RMM_HOME`. 
Follow instructions on this page:
`https://github.com/rapidsai/rmm`, section `Script to build RMM from source`

```xml
$ cd ${RMM_HOME}/
$ ./build.sh librmm
```
NOTE:
1. The build.sh script does installation by default.
2. The destination is conda env `cudf_dev`.

### 3. libcuspatial
Follow the instructions on this page for libcuspatial: https://github.com/rapidsai/cuspatial
```xml
$ cd ${CUSPATIAL_HOME}/
$ ./build.sh libcuspatial tests
```
NOTE:
1. Require env `RMM_HOME`, `CUDF_HOME`, `CUSPATIAL_HOME`, and conda env `cudf_dev`.

## Build cuspatial-java
```xml
$ cd ${CUSPATIAL_HOME}/java
$ mvn clean install
```
NOTE: 
1. cuspatialJni.so will be built by maven pom.
2. Unit tests of Cuspatial-java will run during the JAR build.
