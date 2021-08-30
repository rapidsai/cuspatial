#!/usr/bin/env bash

#Upload cuspatial once per PYTHON
if [[ "$CUDA_VER" == "11.0" ]]; then
    export UPLOAD_CUSPATIAL=1
else
    export UPLOAD_CUSPATIAL=0
fi

#Upload libcuspatial once per CUDA
if [[ "$PYTHON_VER" == "3.7" ]]; then
    export UPLOAD_LIBCUSPATIAL=1
else
    export UPLOAD_LIBCUSPATIAL=0
fi

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    #If project flash is not activate, always build both
    export BUILD_LIBCUSPATIAL=1
    export BUILD_CUSPATIAL=1
fi
