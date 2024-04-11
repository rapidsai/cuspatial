#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels
conda config --system --remove channels rapidsai

source rapids-configure-sccache

source rapids-date-string

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

rapids-logger "Begin py build cuSpatial"

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cuspatial

rapids-logger "Begin py build cuProj"

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cuproj

rapids-upload-conda-to-s3 python
