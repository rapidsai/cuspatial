# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory
export CUDF_HOME="${WORKSPACE}/cudf"
# show environment
printenv
# Cleanup local git
git clean -xdf
# build cudf with verbose output
cd $CUDF_HOME
./build.sh -v libnvstrings nvstrings libcudf cudf
# build cuspatial with verbose output
cd $WORKSPACE
./build.sh -v libcuspatial
