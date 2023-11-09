#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

ARTIFACT="$(realpath "$(dirname "$0")/utils/rapids-pr-artifact-path.sh")"

LIBRMM_CHANNEL=$(${ARTIFACT} rmm 1095 cpp)
LIBCUDF_CHANNEL=$(${ARTIFACT} cudf 14365 cpp)

rapids-print-env

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${LIBCUDF_CHANNEL}" \
    conda/recipes/libcuspatial

rapids-upload-conda-to-s3 cpp
