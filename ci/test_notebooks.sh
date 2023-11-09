#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate notebook testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
PYTHON_MINOR_VERSION=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')

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

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  cuspatial libcuspatial cuproj

NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"
pushd notebooks

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)
SKIPNBS="binary_predicates.ipynb cuproj_benchmark.ipynb"

EXITCODE=0
trap "EXITCODE=1" ERR

set +e
for nb in $(find . -name "*.ipynb"); do
    nbBasename=$(basename ${nb})
    if (echo " ${SKIPNBS} " | grep -q " ${nbBasename} "); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (listed in skip list)"
        echo "--------------------------------------------------------------------------------"
    else
        nvidia-smi
        ${NBTEST} ${nbBasename}
    fi
done

rapids-logger "Notebook test script exiting with value: $EXITCODE"
exit ${EXITCODE}
