#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

export SKBUILD_CMAKE_ARGS="-DUSE_LIBARROW_FROM_PYARROW=ON"

ci/build_wheel.sh cuspatial python/cuspatial
