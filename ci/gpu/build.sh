#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#########################################
# cuSpatial GPU build and test script for CI #
#########################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}
export CUDF_HOME="${WORKSPACE}/cudf"

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install "cudf=${MINOR_VERSION}.*" "cudatoolkit=$CUDA_REL"

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# cuSpatial currently requires a the cudf repo for private headers
################################################################################

logger "Clone cudf"
git clone https://github.com/rapidsai/cudf.git -b branch-$MINOR_VERSION ${CUDF_HOME}
cd $CUDF_HOME
git submodule update --init --remote --recursive

################################################################################
# BUILD - Build libcuspatial and cuSpatial from source
################################################################################

logger "Build cuSpatial"
cd $WORKSPACE
./build.sh clean libcuspatial cuspatial

###############################################################################
# TEST - Run libcuspatial and cuSpatial Unit Tests
###############################################################################

if hasArg --skip-tests; then
    logger "Skipping tests..."
else
    logger "Download/Generate Test Data"
    #TODO

    logger "Test cuSpatial"
    #TODO

    #Python Unit tests for cuSpatial
    cd $WORKSPACE/python/cuspatial
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-cuspatial.xml -v
fi

