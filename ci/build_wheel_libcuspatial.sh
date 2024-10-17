#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

git clone \
  --branch multiple-file-keys \
  https://github.com/jameslamb/dependency-file-generator.git \
  /tmp/rapids-dependency-file-generator

pip uninstall --yes rapids-dependency-file-generator
pip install /tmp/rapids-dependency-file-generator

rapids-logger "Generating build requirements"
matrix_selectors="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "${matrix_selectors}" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
python -m pip install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
export PIP_NO_BUILD_ISOLATION=true

ci/build_wheel.sh libcuspatial python/libcuspatial cpp
