#!/bin/bash

set -e

export LIBCUSPATIAL_FILE=`gpuci_conda_retry build conda/recipes/libcuspatial  --output`
export CUSPATIAL_FILE=`gpuci_conda_retry build conda/recipes/cuspatial --python=$PYTHON --output`

CUDA_REL=${CUDA_VERSION%.*}

if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
fi

if [ "$UPLOAD_LIBCUSPATIAL" == "1" ]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${LIBCUSPATIAL_FILE}
  echo "Upload libcuspatial"
  echo ${LIBCUSPATIAL_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${LIBCUSPATIAL_FILE}
fi

if [ "$UPLOAD_CUSPATIAL" == "1" ]; then
  LABEL_OPTION="--label main"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${CUSPATIAL_FILE}
  echo "Upload cuspatial"
  echo ${CUSPATIAL_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUSPATIAL_FILE}
fi

