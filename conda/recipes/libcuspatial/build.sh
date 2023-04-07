# Copyright (c) 2018-2022, NVIDIA CORPORATION.

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin ${CXX}" # Needed for CUDA 12 nvidia channel compilers

# build cuspatial with verbose output
./build.sh -v libcuspatial tests benchmarks --allgpuarch -n
