#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
package_type=$3

# The 'libcuspatial' wheel should package 'libcuspatial.so', and all others
# should exclude it (they dynamically load it if they need it).
#
# Capturing that here in argument-parsing to allow this build_wheel.sh
# script to be re-used by all wheel builds in the project.
case "${package_dir}" in
  python/libcuspatial)
    EXCLUDE_ARGS=(
      --exclude "libcudf.so"
    )
  ;;
  *)
    EXCLUDE_ARGS=(
      --exclude "libcudf.so"
      --exclude "libcuspatial.so"
    )
  ;;
esac

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

cd "${package_dir}"

rapids-logger "Generating build requirements"
declare -r matrix_selectors="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --matrix "${matrix_selectors}" \
| tee /tmp/requirements-build.txt

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "${matrix_selectors}" \
| tee -a /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
python -m pip install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

rapids-logger "Building '${package_name}' wheel"
python -m pip wheel \
    -w dist \
    -v \
    --no-build-isolation \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w final_dist \
    dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 "${package_type}" final_dist
