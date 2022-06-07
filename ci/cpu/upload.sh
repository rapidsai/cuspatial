#!/bin/bash

set -e

# Setup 'gpuci_retry' for upload retries (results in 4 total attempts)
export GPUCI_RETRY_MAX=3
export GPUCI_RETRY_SLEEP=30

# Set default label options if they are not defined elsewhere
export LABEL_OPTION=${LABEL_OPTION:-"--label main"}

# Skip uploads unless BUILD_MODE == "branch"
if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

# Skip uploads if there is no upload key
if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Starting conda uploads"

if [[ "$BUILD_LIBCUSPATIAL" == "1" && "$UPLOAD_LIBCUSPATIAL" == "1" ]]; then
  LIBCUSPATIAL_FILES=$(conda build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcuspatial --output)
  echo "Upload libcuspatial"
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing --no-progress ${LIBCUSPATIAL_FILES}
fi

if [[ "$BUILD_CUSPATIAL" == "1" && "$UPLOAD_CUSPATIAL" == "1" ]]; then
  CUSPATIAL_FILE=$(conda build --croot ${CONDA_BLD_DIR} conda/recipes/cuspatial --python=$PYTHON --output)
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"
  test -e ${CUSPATIAL_FILE}
  echo "Upload cuspatial"
  echo ${CUSPATIAL_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUSPATIAL_FILE} --no-progress
fi
