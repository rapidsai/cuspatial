#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cuproj"

wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}

ci/build_wheel.sh cuproj ${package_dir} python
ci/validate_wheel.sh ${package_dir} "${wheel_dir}"
