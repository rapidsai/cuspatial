# Copyright (c) 2018-2023, NVIDIA CORPORATION.

# build cuspatial with verbose output
./build.sh -v libcuspatial libcuproj tests benchmarks --allgpuarch -n \
    --cmake-args=\"-DNVBench_ENABLE_CUPTI=OFF\"
