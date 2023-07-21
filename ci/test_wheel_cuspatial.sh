#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eoxu pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/cuspatial*.whl)[test]

# Install additional dependencies
apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libgdal-dev && python -m pip install --no-binary fiona 'fiona>=1.8.19,<1.9'

arch=$(uname -m)
if [[ "${arch}" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test.py
else
    python -m pytest -n 8 ./python/cuspatial/cuspatial/tests
fi
