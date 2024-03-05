#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eou pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cuspatial_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# Install additional dependencies
apt update
DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libgdal-dev
python -m pip install --no-binary fiona 'fiona>=1.8.19,<1.9'

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/cuspatial*.whl)[test]

if [[ "$(arch)" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test_cuspatial.py
else
    rapids-logger "pytest cuspatial"
    pushd python/cuspatial/cuspatial
    python -m pytest \
      --cache-clear \
      --junitxml="${RAPIDS_TESTS_DIR}/junit-cuproj.xml" \
      --numprocesses=8 \
      --dist=worksteal \
      tests
    popd
fi
