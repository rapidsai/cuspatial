#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

ARTIFACT="$(realpath "$(dirname "$0")/utils/rapids-get-pr-artifact.sh")"

LIBRMM_CHANNEL=$(${ARTIFACT} rmm 1095 cpp)
RMM_CHANNEL=$(${ARTIFACT} rmm 1095 python)
LIBCUDF_CHANNEL=$(${ARTIFACT} cudf 14365 cpp)
CUDF_CHANNEL=$(${ARTIFACT} cudf 14365 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

# CUSPATIAL_HOME is used to find test files
export CUSPATIAL_HOME="${PWD}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  libcuspatial cuspatial cuproj

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuspatial"
pushd python/cuspatial/cuspatial
# It is essential to cd into python/cuspatial/cuspatial as `pytest-xdist` + `coverage` seem to work only at this directory level.
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuspatial.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=../.coveragerc \
  --cov=cuspatial \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuspatial-coverage.xml" \
  --cov-report=term \
  tests
popd

rapids-logger "pytest cuproj"
pushd python/cuproj/cuproj
# It is essential to cd into python/cuproj/cuproj as `pytest-xdist` + `coverage` seem to work only at this directory level.
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuproj.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=../.coveragerc \
  --cov=cuproj \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuproj-coverage.xml" \
  --cov-report=term \
  tests
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
