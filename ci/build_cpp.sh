#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1404 cpp)
LIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 14576 cpp)

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${LIBCUDF_CHANNEL}" \
    conda/recipes/libcuspatial

rapids-upload-conda-to-s3 cpp
