#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

source ./ci/use_wheels_from_prs.sh

# Download the cuproj and cuspatial built in the previous step
RAPIDS_PY_WHEEL_NAME="cuproj_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="cuspatial_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="libcuspatial_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
  "$(echo ./dist/cuspatial*.whl)" \
  "$(echo ./dist/cuproj*.whl)[test]" \
  "$(echo ./dist/libcuspatial*.whl)"

rapids-logger "pytest cuproj"
pushd python/cuproj/cuproj
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuproj.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  tests
popd
