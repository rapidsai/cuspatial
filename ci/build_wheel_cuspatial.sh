#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

export SKBUILD_CONFIGURE_OPTIONS="-DCUSPATIAL_BUILD_WHEELS=ON"

ci/build_wheel.sh cuspatial python/cuspatial
