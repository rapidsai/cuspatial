#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# install build dependencies for fiona
if type -f yum > /dev/null 2>&1; then
  yum update -y
  # some of gdal-devel's dependencies, like 'libdap', come from the powertools repo
  yum config-manager --set-enabled powertools
  yum update -y
  yum install -y gdal-devel
else
  apt update
  DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libgdal-dev
fi

# Download the cuproj and cuspatial built in the previous step
RAPIDS_PY_WHEEL_NAME="cuproj_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="cuspatial_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
  --no-binary 'fiona' \
  "$(echo ./dist/cuspatial*.whl)" \
  "$(echo ./dist/cuproj*.whl)[test]" \
  'fiona>=1.8.19,<1.9'

rapids-logger "pytest cuproj"
pushd python/cuproj/cuproj
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuproj.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  tests
popd
