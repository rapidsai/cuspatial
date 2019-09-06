#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# cuSpatial CPU conda build script for CI #
######################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDF_HOME="${WORKSPACE}/cudf"

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

##########################################################################################
# BUILD - Conda package builds (conda deps: libcupatial <- libcuspatial_cffi <- cuspatial)
##########################################################################################

logger "Clone cudf"
git clone https://github.com/rapidsai/cudf.git -b branch-$MINOR_VERSION $CUDF_HOME

logger "Build conda pkg for libcuspatial..."
source ci/cpu/libcuspatial/build_libcuspatial.sh

logger "Build conda pkg for cuspatial..."
source ci/cpu/cuspatial/build_cuspatial.sh

################################################################################
# UPLOAD - Conda packages
################################################################################

logger "Upload conda pkgs..."
source ci/cpu/upload_anaconda.sh

