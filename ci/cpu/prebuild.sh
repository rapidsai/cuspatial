#!/usr/bin/env bash

#Upload cuspatial once per CUDA
if [[ "$CUDA" == "9.2" ]]; then
    export UPLOAD_CUDF=1
else
    export UPLOAD_CUDF=0
fi

#Upload libcuspatial once per PYTHON
if [[ "$PYTHON" == "3.6" ]]; then
    export UPLOAD_LIBCUDF=1
else
    export UPLOAD_LIBCUDF=0
fi
