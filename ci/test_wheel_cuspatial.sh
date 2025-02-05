#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eou pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the cuspatial and libcuspatial built in the previous step
RAPIDS_PY_WHEEL_NAME="cuspatial_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="libcuspatial_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
  "$(echo ./dist/cuspatial*.whl)[test]" \
  "$(echo ./dist/libcuspatial*.whl)"

rapids-logger "pytest cuspatial"
pushd python/cuspatial/cuspatial
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuspatial.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  tests
popd
