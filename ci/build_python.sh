#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_dir="python"

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

echo "${version}" > VERSION
for package_name in cuspatial cuproj; do
    sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_dir}/${package_name}/${package_name}/_version.py"
done

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

rapids-logger "Begin py build cuSpatial"

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  conda/recipes/cuspatial

rapids-logger "Begin py build cuProj"

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  conda/recipes/cuproj

rapids-upload-conda-to-s3 python
