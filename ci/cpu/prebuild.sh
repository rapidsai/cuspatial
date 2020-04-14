#!/usr/bin/env bash

#Upload cuspatial once per PYTHON
if [[ "$CUDA" == "10.0" ]]; then
    export UPLOAD_CUSPATIAL=1
else
    export UPLOAD_CUSPATIAL=0
fi

#Upload libcuspatial once per CUDA
if [[ "$PYTHON" == "3.6" ]]; then
    export UPLOAD_LIBCUSPATIAL=1
else
    export UPLOAD_LIBCUSPATIAL=0
fi

