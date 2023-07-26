#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/cuspatial/pyproject.toml
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/cuproj/pyproject.toml

sed -i "s/^name = \"cuspatial\"/name = \"cuspatial${CUDA_SUFFIX}\"/g" python/cuspatial/pyproject.toml
sed -i "s/^name = \"cuproj\"/name = \"cuproj${CUDA_SUFFIX}\"/g" python/cuproj/pyproject.toml

sed -i "s/rmm==/rmm${CUDA_SUFFIX}==/g" python/cuspatial/pyproject.toml
sed -i "s/cudf==/cudf${CUDA_SUFFIX}==/g" python/cuspatial/pyproject.toml

sed -i "s/rmm==/rmm${CUDA_SUFFIX}==/g" python/cuproj/pyproject.toml
sed -i "s/cudf==/cudf${CUDA_SUFFIX}==/g" python/cuproj/pyproject.toml
