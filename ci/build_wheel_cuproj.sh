#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cuproj"

ci/build_wheel.sh cuproj ${package_dir} python
ci/validate_wheel.sh ${package_dir} final_dist
