#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
PYTHON_MINOR_VERSION=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids_repo_pr_artifact_channel () {
    local repo=$1
    local pr=$2
    local commit=$(git ls-remote https://github.com/rapidsai/${repo}.git refs/heads/pull-request/${pr} | cut -c1-7)

    if [[ $3 == "cpp" ]] then
        echo $(rapids-get-artifact ci/${repo}/pull-request/${pr}/${commit}/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
    else
        echo $(rapids-get-artifact ci/${repo}/pull-request/${pr}/${commit}/rmm_conda_python_cuda${RAPIDS_CUDA_MAJOR}_3${PYTHON_MINOR_VERSION}_$(arch).tar.gz)
    fi
}

LIBRMM_CHANNEL=$(rapids_repo_pr_artifact_channel rmm 1095 cpp)
RMM_CHANNEL=$(rapids_repo_pr_artifact_channel rmm 1095 python)
LIBCUDF_CHANNEL=$(rapids_repo_pr_artifact_channel cudf 14365 cpp)
CUDF_CHANNEL=$(rapids_repo_pr_artifact_channel cudf 14365 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# CUSPATIAL_HOME is used to find test files
export CUSPATIAL_HOME="${PWD}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  libcuspatial libcuspatial-tests

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run libcuspatial gtests from libcuspatial-tests package
rapids-logger "Run gtests"
for gt in "$CONDA_PREFIX"/bin/gtests/libcuspatial/* ; do
    test_name=$(basename ${gt})
    echo "Running gtest $test_name"
    ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}
done

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
