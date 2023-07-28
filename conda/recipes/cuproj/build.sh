
# Copyright (c) 2018-2023, NVIDIA CORPORATION.

# Ignore conda-provided CMAKE_ARGS for the Python build.
unset CMAKE_ARGS

# This assumes the script is executed from the root of the repo directory
./build.sh cuproj
