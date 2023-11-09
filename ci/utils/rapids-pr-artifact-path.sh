#!/bin/bash
# Echos path to an artifact for a specific PR. It uses the latest commit on the PR.
#
# Positional Arguments:
#   1) repo name
#   2) PR number
#   3) cpp or python
#
#
# Example Usage:
#   rapids-pr-artifact-path rmm 1095 cpp

set -euo pipefail

repo="$1"
pr="$2"
commit=$(git ls-remote https://github.com/rapidsai/${repo}.git refs/heads/pull-request/${pr} | cut -c1-7)

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
PYTHON_MINOR_VERSION=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')

if [[ $3 == "cpp" ]]
then
    echo $(rapids-get-artifact ci/${repo}/pull-request/${pr}/${commit}/${repo}_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
else
    echo $(rapids-get-artifact ci/${repo}/pull-request/${pr}/${commit}/${repo}_conda_python_cuda${RAPIDS_CUDA_MAJOR}_3${PYTHON_MINOR_VERSION}_$(arch).tar.gz)
fi
