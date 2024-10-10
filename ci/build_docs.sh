#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL="$(rapids-download-conda-from-s3 cpp)"
PYTHON_CHANNEL="$(rapids-download-conda-from-s3 python)"

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" --prepend-channel "${PYTHON_CHANNEL}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

export RAPIDS_VERSION="$(rapids-version)"
export RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build cuSpatial CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libcuspatial/html"
mv html/* "${RAPIDS_DOCS_DIR}/libcuspatial/html"
popd

rapids-logger "Build cuProj CPP docs"
pushd cpp/cuproj/doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libcuproj/html"
mv html/* "${RAPIDS_DOCS_DIR}/libcuproj/html"
popd

rapids-logger "Build cuSpatial Python docs"
pushd docs
sphinx-build -b dirhtml source _html -W
mkdir -p "${RAPIDS_DOCS_DIR}/cuspatial/html"
mv _html/* "${RAPIDS_DOCS_DIR}/cuspatial/html"
popd

rapids-logger "Build cuProj Python docs"
pushd docs/cuproj
sphinx-build -b dirhtml source _html -W
mkdir -p "${RAPIDS_DOCS_DIR}/cuproj/html"
mv _html/* "${RAPIDS_DOCS_DIR}/cuproj/html"
popd

RAPIDS_VERSION_NUMBER="${RAPIDS_VERSION_MAJOR_MINOR}" rapids-upload-docs
