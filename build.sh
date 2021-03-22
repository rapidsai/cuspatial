#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.

# cuSpatial build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean libcuspatial cuspatial tests -v -g -n -h --allgpuarch --show_depr_warn"
HELP="$0 [clean] [libcuspatial] [cuspatial] [tests] [-v] [-g] [-n] [-h] [-l] [--show_depr_warn]
   clean            - remove all existing build artifacts and configuration (start
                      over)
   libcuspatial     - build the libcuspatial C++ code only
   cuspatial        - build the cuspatial Python package
   tests            - build tests
   -v               - verbose build mode
   -g               - build for debug
   -n               - no install step
   -h               - print this text
   --allgpuarch     - build for all supported GPU architectures
   --show_depr_warn - show cmake deprecation warnings
   default action (no args) is to build and install 'libcuspatial' then
   'cuspatial' targets
"
LIBCUSPATIAL_BUILD_DIR=${REPODIR}/cpp/build
CUSPATIAL_BUILD_DIR=${REPODIR}/python/cuspatial/build
BUILD_DIRS="${LIBCUSPATIAL_BUILD_DIR} ${CUSPATIAL_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TESTS=OFF
BUILD_TYPE=Release
BUILD_ALL_GPU_ARCH=0
INSTALL_TARGET=install
BUILD_DISABLE_DEPRECATION_WARNING=ON

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG="-v"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNING=OFF
fi

if hasArg tests; then
    BUILD_TESTS=ON
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
    if [ -d ${bd} ]; then
        find ${bd} -mindepth 1 -delete
        rmdir ${bd} || true
    fi
    done
fi

if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
    CUSPATIAL_CMAKE_CUDA_ARCHITECTURES="-DCMAKE_CUDA_ARCHITECTURES="
    echo "Building for the architecture of the GPU in the system..."
else
    CUSPATIAL_CMAKE_CUDA_ARCHITECTURES=""
    echo "Building for *ALL* supported GPU architectures..."
fi

################################################################################
# Configure, build, and install libcuspatial
if (( ${NUMARGS} == 0 )) || hasArg libcuspatial; then
    mkdir -p ${LIBCUSPATIAL_BUILD_DIR}
    cd ${LIBCUSPATIAL_BUILD_DIR}
    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          ${CUSPATIAL_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_CXX11_ABI=ON \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          ..

    cmake --build . -j ${PARALLEL_LEVEL} ${VERBOSE_FLAG}

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j ${PARALLEL_LEVEL} --target install ${VERBOSE_FLAG}
    fi
fi

# Build and install the cuspatial Python package
if (( ${NUMARGS} == 0 )) || hasArg cuspatial; then

    cd ${REPODIR}/python/cuspatial
    if [[ ${INSTALL_TARGET} != "" ]]; then
        PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace
        python setup.py install --single-version-externally-managed --record=record.txt
    else
        PARALLEL_LEVEL=${PARALLEL_LEVEL} python setup.py build_ext --inplace --library-dir=${LIBCUSPATIAL_BUILD_DIR}
    fi
fi
