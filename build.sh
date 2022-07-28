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

VALIDARGS="clean libcuspatial cuspatial tests benchmarks -v -g -n -h --allgpuarch --show_depr_warn"
HELP="$0 [clean] [libcuspatial] [cuspatial] [tests] [-v] [-g] [-n] [-h] [-l] [--show_depr_warn] [--cmake-args=\"<args>\"]
   clean                       - remove all existing build artifacts and configuration (start over)
   libcuspatial                - build the libcuspatial C++ code only
   cuspatial                   - build the cuspatial Python package
   tests                       - build tests
   benchmarks                  - build benchmarks
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step
   -h                          - print this text
   --allgpuarch                - build for all supported GPU architectures
   --show_depr_warn            - show cmake deprecation warnings
   --cmake-args=\\\"<args>\\\" - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   default action (no args) is to build and install 'libcuspatial' then
   'cuspatial' targets
"
LIBCUSPATIAL_BUILD_DIR=${REPODIR}/cpp/build
CUSPATIAL_BUILD_DIR=${REPODIR}/python/cuspatial/_skbuild
BUILD_DIRS="${LIBCUSPATIAL_BUILD_DIR} ${CUSPATIAL_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TESTS=OFF
BUILD_BENCHMARKS=OFF
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

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo $ARGS | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo $EXTRA_CMAKE_ARGS | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
}

if hasArg -h; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    cmakeArgs
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

if hasArg benchmarks; then
    BUILD_BENCHMARKS=ON
fi

# Append `-DFIND_CUSPATIAL_CPP=ON` to EXTRA_CMAKE_ARGS unless a user specified the option.
if [[ "${EXTRA_CMAKE_ARGS}" != *"DFIND_CUSPATIAL_CPP"* ]]; then
    EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DFIND_CUSPATIAL_CPP=ON"
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
    CUSPATIAL_CMAKE_CUDA_ARCHITECTURES="-DCMAKE_CUDA_ARCHITECTURES=NATIVE"
    echo "Building for the architecture of the GPU in the system..."
else
    CUSPATIAL_CMAKE_CUDA_ARCHITECTURES="-DCMAKE_CUDA_ARCHITECTURES=ALL"
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
          -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          ${EXTRA_CMAKE_ARGS} \
          ..

    cmake --build . -j ${PARALLEL_LEVEL} ${VERBOSE_FLAG}

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j ${PARALLEL_LEVEL} --target install ${VERBOSE_FLAG}
    fi
fi

# Build and install the cuspatial Python package
if (( ${NUMARGS} == 0 )) || hasArg cuspatial; then

    cd ${REPODIR}/python/cuspatial
    python setup.py build_ext -j${PARALLEL_LEVEL:-1} --inplace -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_LIBRARY_PATH=${LIBCUSPATIAL_BUILD_DIR} ${EXTRA_CMAKE_ARGS}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py install --single-version-externally-managed --record=record.txt -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_LIBRARY_PATH=${LIBCUSPATIAL_BUILD_DIR} ${EXTRA_CMAKE_ARGS}
    fi
fi
