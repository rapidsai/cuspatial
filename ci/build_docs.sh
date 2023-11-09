#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n docs
conda activate docs

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

    if [[ $3 == "cpp" ]]
    then
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
  libcuspatial \
  cuspatial \
  cuproj

export RAPIDS_VERSION_NUMBER="23.12"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build cuSpatial CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libcuspatial/html"
mv html/* "${RAPIDS_DOCS_DIR}/libcuspatial/html"
popd

rapids-logger "Build cuProj CPP docs"
pushd cpp/cuproj/doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libcuproj/html"
mv html/* "${RAPIDS_DOCS_DIR}/libcuproj/html"
popd

rapids-logger "Build cuSpatial Python docs"
pushd docs
sphinx-build -b dirhtml source _html -W
sphinx-build -b text source _text -W
mkdir -p "${RAPIDS_DOCS_DIR}/cuspatial/"{html,txt}
mv _html/* "${RAPIDS_DOCS_DIR}/cuspatial/html"
mv _text/* "${RAPIDS_DOCS_DIR}/cuspatial/txt"
popd

rapids-logger "Build cuProj Python docs"
pushd docs/cuproj
sphinx-build -b dirhtml source _html -W
sphinx-build -b text source _text -W
mkdir -p "${RAPIDS_DOCS_DIR}/cuproj/"{html,txt}
mv _html/* "${RAPIDS_DOCS_DIR}/cuproj/html"
mv _text/* "${RAPIDS_DOCS_DIR}/cuproj/txt"
popd

rapids-upload-docs
