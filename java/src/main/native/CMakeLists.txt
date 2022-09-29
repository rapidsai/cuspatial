#=============================================================================
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(../../../../fetch_rapids.cmake)

# TODO: The logic for setting the architectures was previously not here. Was
# that just an oversight, or is there a reason not to include this here?
rapids_cuda_init_architectures(CUSPATIAL_JNI)

project(CUDF_JNI VERSION 0.7.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type("Release")

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated-declarations")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "CUDF: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "CUDF: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# set warnings as errors
# TODO: remove `no-maybe-unitialized` used to suppress warnings in rmm::exec_policy
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror,-Wno-error=deprecated-declarations")

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
if (CMAKE_CUDA_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler -rdynamic")
endif()

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

###################################################################################################
# - cudart options --------------------------------------------------------------------------------
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking

option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

if(CUDA_STATIC_RUNTIME)
    message(STATUS "Enabling static linking of cudart")
    set(CUDART_LIBRARY "cudart_static")
else()
    set(CUDART_LIBRARY "cudart")
endif()

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - Thrust/CUB/libcudacxx ------------------------------------------------------------------------------------

find_path(LIBCUDACXX_INCLUDE "simt"
    HINTS "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx"
          "$ENV{CUDF_HOME}/cpp/build/_deps/libcudacxx-src/include")

message(STATUS "CUDACXX: LIBCUDACXX_INCLUDE set to ${LIBCUDACXX_INCLUDE}")

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{CONDA_PREFIX}/include"
                "$ENV{CONDA_PREFIX}/include/rmm"
                "$ENV{RMM_HOME}/include")

find_library(RMM_LIBRARY "rmm"
             HINTS "$ENV{CONDA_PREFIX}/lib"
                   "$ENV{RMM_HOME}/build")

message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")
message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if (RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif (RMM_INCLUDE AND RMM_LIBRARY)

###################################################################################################
# - CUDF ------------------------------------------------------------------------------------------

set(CUDF_INCLUDE "$ENV{CONDA_PREFIX}/include"
                 "$ENV{CUDF_HOME}/java/src/main/native/include")

find_library(CUDF_LIBRARY "cudf"
    HINTS "$ENV{CONDA_PREFIX}/lib"
          "$ENV{CUDF_HOME}/cpp/build")

message(STATUS "CUDF: CUDF_INCLUDE set to ${CUDF_INCLUDE}")
message(STATUS "CUDF: CUDF_LIBRARY set to ${CUDF_LIBRARY}")

add_library(cudf SHARED IMPORTED ${CUDF_LIBRARY})
if (CUDF_INCLUDE AND CUDF_LIBRARY)
    set_target_properties(cudf PROPERTIES IMPORTED_LOCATION ${CUDF_LIBRARY})
endif (CUDF_INCLUDE AND CUDF_LIBRARY)

###################################################################################################
# - CUSPATIAL ------------------------------------------------------------------------------------------

set(CUSPATIAL_INCLUDE "$ENV{CUSPATIAL_HOME}/cpp/include")

find_library(CUSPATIAL_LIBRARY "cuspatial"
    HINTS "$ENV{CUSPATIAL_HOME}/cpp/build")

message(STATUS "CUSPATIAL: CUSPATIAL_INCLUDE set to ${CUSPATIAL_INCLUDE}")
message(STATUS "CUSPATIAL: CUSPATIAL_LIBRARY set to ${CUSPATIAL_LIBRARY}")

add_library(cuspatial SHARED IMPORTED ${CUSPATIAL_LIBRARY})
if (CUSPATIAL_INCLUDE AND CUSPATIAL_LIBRARY)
    set_target_properties(cuspatial PROPERTIES IMPORTED_LOCATION ${CUSPATIAL_LIBRARY})
endif (CUSPATIAL_INCLUDE AND CUSPATIAL_LIBRARY)

###################################################################################################
# - find JNI -------------------------------------------------------------------------------------
find_package(JNI REQUIRED)
if(JNI_FOUND)
    message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif()

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${LIBCUDACXX_INCLUDE}"
                    "${RMM_INCLUDE}"
                    "${CUDF_INCLUDE}"
                    "${CUSPATIAL_INCLUDE}"
                    "${JNI_INCLUDE_DIRS}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_BINARY_DIR}/lib"
                 "${CUSPATIAL_LIBRARY}")


###################################################################################################
# - library targets -------------------------------------------------------------------------------

set(SOURCE_FILES
    "src/cuSpatialJni.cpp")
add_library(cuspatialjni SHARED ${SOURCE_FILES})

#Override RPATH for cuspatialjni
SET_TARGET_PROPERTIES(cuspatialjni PROPERTIES BUILD_RPATH "\$ORIGIN")

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(cuspatialjni ${NVTX_LIBRARY})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX")
endif()

option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
if(PER_THREAD_DEFAULT_STREAM)
    message(STATUS "Using per-thread default stream")
    add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(cuspatialjni cuspatial)
