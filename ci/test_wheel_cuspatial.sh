#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# install build dependencies for fiona
apt update
DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libgdal-dev

# Download the cuspatial built in the previous step
RAPIDS_PY_WHEEL_NAME="cuspatial_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
  --no-binary 'fiona' \
  "$(echo ./dist/cuspatial*.whl)[test]" \
  'fiona>=1.8.19,<1.9'

rapids-logger "pytest cuspatial"
pushd python/cuspatial/cuspatial
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuspatial.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  tests
popd
