#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

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
      --exclude "librapids_logger.so"
    )
  ;;
  *)
    EXCLUDE_ARGS=(
      --exclude "libcudf.so"
      --exclude "libcuspatial.so"
      --exclude "librapids_logger.so"
    )
  ;;
esac

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

cd "${package_dir}"

rapids-logger "Building '${package_name}' wheel"
rapids-pip-retry wheel \
    -w dist \
    -v \
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
