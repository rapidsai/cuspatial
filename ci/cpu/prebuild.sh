#!/usr/bin/env bash

export UPLOAD_CUSPATIAL=1
export UPLOAD_LIBCUSPATIAL=1

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    #If project flash is not activate, always build both
    export BUILD_LIBCUSPATIAL=1
    export BUILD_CUSPATIAL=1
fi
