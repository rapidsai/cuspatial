#!/bin/bash
# Echo path to an artifact for a specific PR. Finds and uses the latest commit on the PR.
#
# Positional Arguments:
#   1) repo name
#   2) PR number
#   3) "cpp" or "python", to get the artifact for the C++ or Python build, respectively
#
# Example Usage:
#   rapids-pr-artifact-path rmm 1095 cpp

set -euo pipefail

repo="$1"
pr="$2"
commit=$(git ls-remote https://github.com/rapidsai/${repo}.git refs/heads/pull-request/${pr} | cut -c1-7)

rapids_cuda_major="${RAPIDS_CUDA_VERSION%%.*}"
python_minor=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')

if [[ $3 == "cpp" ]]
then
    echo $(rapids-get-artifact ci/${repo}/pull-request/${pr}/${commit}/${repo}_conda_cpp_cuda${rapids_cuda_major}_$(arch).tar.gz)
else
    echo $(rapids-get-artifact ci/${repo}/pull-request/${pr}/${commit}/${repo}_conda_python_cuda${rapids_cuda_major}_3${python_minor}_$(arch).tar.gz)
fi
