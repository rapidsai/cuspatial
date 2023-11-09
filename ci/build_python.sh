#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_dir="python"

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

echo "${version}" > VERSION
for package_name in cuspatial cuproj; do
    sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_dir}/${package_name}/${package_name}/_version.py"
done

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

ARTIFACT="$(realpath "$(dirname "$0")/utils/rapids-get-pr-artifact.sh")"

LIBRMM_CHANNEL=$(${ARTIFACT} rmm 1095 cpp)
RMM_CHANNEL=$(${ARTIFACT} rmm 1095 python)
LIBCUDF_CHANNEL=$(${ARTIFACT} cudf 14365 cpp)
CUDF_CHANNEL=$(${ARTIFACT} cudf 14365 python)

rapids-logger "Begin py build cuSpatial"

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  conda/recipes/cuspatial

rapids-logger "Begin py build cuProj"

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  conda/recipes/cuproj

rapids-upload-conda-to-s3 python
