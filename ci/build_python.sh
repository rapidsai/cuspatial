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

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

REPO="rmm"
PR_NUMBER="1095"
COMMIT=$(git ls-remote https://github.com/rapidsai/${REPO}.git refs/heads/pull-request/${PR_NUMBER} | cut -c1-7)
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
PYTHON_MINOR_VERSION=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')
LIBRMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
RMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_python_cuda${RAPIDS_CUDA_MAJOR}_3${PYTHON_MINOR_VERSION}_$(arch).tar.gz)

rapids-logger "Begin py build cuSpatial"

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  conda/recipes/cuspatial

rapids-logger "Begin py build cuProj"

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  conda/recipes/cuproj

rapids-upload-conda-to-s3 python
