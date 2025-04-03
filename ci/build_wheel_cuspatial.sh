#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cuspatial"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcuspatial wheel from this current build,
# then ensures 'cuspatial' wheel builds always use the 'libcuspatial' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
RAPIDS_PY_WHEEL_NAME="libcuspatial_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcuspatial_dist
echo "libcuspatial-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcuspatial_dist/libcuspatial_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

ci/build_wheel.sh cuspatial ${package_dir} python
ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
