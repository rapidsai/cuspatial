#!/bin/bash
# Copyright (c) 2020-2024, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-logger "Generate notebook testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" --prepend-channel "${PYTHON_CHANNEL}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)
SKIPNBS="binary_predicates.ipynb cuproj_benchmark.ipynb"

EXITCODE=0
trap "EXITCODE=1" ERR

set +e

test_notebooks() {
    for nb in $(find . -name "*.ipynb"); do
        nbBasename=$(basename ${nb})
        if (echo " ${SKIPNBS} " | grep -q " ${nbBasename} "); then
            echo "--------------------------------------------------------------------------------"
            echo "SKIPPING: ${nb} (listed in skip list)"
            echo "--------------------------------------------------------------------------------"
        else
            nvidia-smi
            ${NBTEST} ${nbBasename}
        fi
    done
}

# test notebooks in notebooks/
pushd notebooks
test_notebooks
popd

# test notebooks in docs/
pushd docs
test_notebooks
popd

rapids-logger "Notebook test script exiting with value: $EXITCODE"
exit ${EXITCODE}
