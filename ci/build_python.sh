#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-logger "Begin py build cuSpatial"

sccache --zero-stats

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cuspatial

sccache --show-adv-stats

rapids-logger "Begin py build cuProj"

sccache --zero-stats

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cuproj

sccache --show-adv-stats

rapids-upload-conda-to-s3 python
