#!/bin/bash
# COPYRIGHT (c) 2020, NVIDIA CORPORATION.
#########################################
# cuSpatial GPU build and test script for CI #
#########################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}
export CUDF_HOME="${WORKSPACE}/cudf"
export CUSPATIAL_HOME="${WORKSPACE}"

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids
gpuci_conda_retry install "cudf=${MINOR_VERSION}.*" "cudatoolkit=$CUDA_REL" \
    "rapids-build-env=$MINOR_VERSION.*"

# https://docs.rapids.ai/maintainers/depmgmt/ 
# gpuci_conda_retry remove --force rapids-build-env
# gpuci_conda_retry install "your-pkg=1.0.0"

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# cuSpatial currently requires a the cudf repo for private headers
################################################################################

gpuci_logger "Clone cudf"
git clone https://github.com/rapidsai/cudf.git -b branch-$MINOR_VERSION ${CUDF_HOME}
cd $CUDF_HOME
git submodule update --init --remote --recursive

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    ################################################################################
    # BUILD - Build libcuspatial and cuSpatial from source
    ################################################################################

    gpuci_logger "Build cuSpatial"
    cd $WORKSPACE
    ./build.sh clean libcuspatial cuspatial tests

    ###############################################################################
    # TEST - Run libcuspatial and cuSpatial Unit Tests
    ###############################################################################

    if hasArg --skip-tests; then
        gpuci_logger "Skipping tests"
    else
        gpuci_logger "Check GPU usage"
        nvidia-smi

        gpuci_logger "GoogleTests"
        cd $WORKSPACE/cpp/build

        for gt in ${WORKSPACE}/cpp/build/gtests/* ; do
            test_name=$(basename ${gt})
            echo "Running GoogleTest $test_name"
            ${gt} --gtest_output=xml:${WORKSPACE}/test-results/
        done

        gpuci_logger "Download/Generate Test Data"
        #TODO

        gpuci_logger "Test cuSpatial"
        #TODO

        #Python Unit tests for cuSpatial
        cd $WORKSPACE/python/cuspatial
        py.test --cache-clear --junitxml=${WORKSPACE}/junit-cuspatial.xml -v
    fi
else
    export LD_LIBRARY_PATH="$WORKSPACE/ci/artifacts/cuspatial/cpu/conda_work/cpp/build:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    TESTRESULTS_DIR=${WORKSPACE}/test-results/
    mkdir -p ${TESTRESULTS_DIR}
    SUITEERROR=0

    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "Running googletests"
    # run gtests
    cd $WORKSPACE/ci/artifacts/cuspatial/cpu/conda_work/
    for gt in cpp/build/gtests/* ; do
        test_name=$(basename ${gt})
        echo "Running GoogleTest $test_name"
        ${gt} --gtest_output=xml:${TESTRESULTS_DIR}
        EXITCODE=$?
        if (( ${EXITCODE} != 0 )); then
            SUITEERROR=${EXITCODE}
            echo "FAILED: GTest ${gt}"
        fi
    done

    cd $WORKSPACE/python
    
    CONDA_FILE=`find $WORKSPACE/ci/artifacts/cuspatial/cpu/conda-bld/ -name "libcuspatial*.tar.bz2"`
    CONDA_FILE=`basename "$CONDA_FILE" .tar.bz2` #get filename without extension
    CONDA_FILE=${CONDA_FILE//-/=} #convert to conda install
    gpuci_logger "Installing $CONDA_FILE"
    conda install -c $WORKSPACE/ci/artifacts/cuspatial/cpu/conda-bld/ "$CONDA_FILE"

    export LIBCUGRAPH_BUILD_DIR="$WORKSPACE/ci/artifacts/cuspatial/cpu/conda_work/build"
    
    gpuci_logger "Building cuspatial"
    "$WORKSPACE/build.sh" -v cuspatial
    
    gpuci_logger "Run pytests"
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-cuspatial.xml -v

    EXITCODE=$?
    if (( ${EXITCODE} != 0 )); then
        SUITEERROR=${EXITCODE}
        echo "FAILED: 1 or more tests in /cuspatial/python"
    fi
    gpuci_logger "Download/Generate Test Data"
    #TODO

    gpuci_logger "Test cuSpatial"
    #TODO

    exit ${SUITEERROR}
fi

