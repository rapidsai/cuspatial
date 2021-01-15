#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# cuSpatial CPU conda build script for CI #
######################################
set -e

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDF_HOME="${WORKSPACE}/cudf"

export GIT_DESCRIBE=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE | grep -o -E '([0-9]+\.[0-9]+)'`

# Set home to the job's workspace
export HOME=$WORKSPACE

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

##########################################################################################
# BUILD - Conda package builds (conda deps: libcupatial <- cuspatial)
##########################################################################################

if [ "$BUILD_LIBCUSPATIAL" == '1' ]; then
  gpuci_logger "Build conda pkg for libcuspatial"
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry build conda/recipes/libcuspatial
  else
    gpuci_conda_retry build --dirty --no-remove-work-dir conda/recipes/libcuspatial
  fi
fi

if [ "$BUILD_CUSPATIAL" == '1' ]; then
  gpuci_logger "Build conda pkg for cuspatial"
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry build conda/recipes/cuspatial
  else
    gpuci_conda_retry build --dirty --no-remove-work-dir \
        -c $WORKSPACE/ci/artifacts/cuspatial/cpu/conda-bld/ conda/recipes/cuspatial
  fi
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload conda pkgs..."
source ci/cpu/upload.sh
