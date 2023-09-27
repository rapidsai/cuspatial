#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

export SKBUILD_CONFIGURE_OPTIONS="-DCUPROJ_BUILD_WHEELS=ON"

ci/build_wheel.sh cuproj python/cuproj
