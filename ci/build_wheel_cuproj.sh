#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

ci/build_wheel.sh cuproj python/cuproj python
