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
# SETUP - Get conda file output locations
################################################################################

gpuci_logger "Get conda file output locations"
export LIBCUSPATIAL_FILE=`conda build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcuspatial  --output`
export CUSPATIAL_FILE=`conda build --croot ${CONDA_BLD_DIR} conda/recipes/cuspatial --python=$PYTHON --output`

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Starting conda uploads"

if [[ "$BUILD_LIBCUSPATIAL" == "1" && "$UPLOAD_LIBCUSPATIAL" == "1" ]]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"
  test -e ${LIBCUSPATIAL_FILE}
  echo "Upload libcuspatial"
  echo ${LIBCUSPATIAL_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${LIBCUSPATIAL_FILE} --no-progress
fi

if [[ "$BUILD_CUSPATIAL" == "1" && "$UPLOAD_CUSPATIAL" == "1" ]]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"
  test -e ${CUSPATIAL_FILE}
  echo "Upload cuspatial"
  echo ${CUSPATIAL_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUSPATIAL_FILE} --no-progress
fi
