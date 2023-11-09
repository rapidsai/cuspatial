#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

REPO="rmm"
PR_NUMBER="1095"
COMMIT=$(git ls-remote https://github.com/rapidsai/${REPO}.git refs/heads/pull-request/${PR_NUMBER} | cut -c1-7)
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
LIBRMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)

rapids-print-env

rapids-logger "Begin cpp build"

rapids-conda-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    conda/recipes/libcuspatial

rapids-upload-conda-to-s3 cpp
