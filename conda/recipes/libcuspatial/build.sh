# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory
# show environment
printenv
# build cudf with verbose output
cd $CUDF_HOME
./build.sh -v libnvstrings nvstrings libcudf cudf
# build cuspatial with verbose output
cd $WORKSPACE
./build.sh -v libcuspatial
