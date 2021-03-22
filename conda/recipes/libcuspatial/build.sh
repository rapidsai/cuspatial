# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# build cuspatial with verbose output
if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    ./build.sh -v libcuspatial --allgpuarch
else
    ./build.sh -v libcuspatial tests --allgpuarch
fi
