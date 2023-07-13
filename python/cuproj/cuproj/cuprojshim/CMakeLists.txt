# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

set(cuproj_version 23.08.00)

include(../../../../fetch_rapids.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(cuproj-shim)

project(
  cuprojshim
  VERSION ${cuproj_version}
  LANGUAGES CXX CUDA)


###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)

message(STATUS "CUPROJSHIM: Build with NVTX support: ${USE_NVTX}")
message(STATUS "CUPROJSHIM: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(STATUS "CUPROJSHIM: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNING}")
message(STATUS "CUPROJSHIM: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler: ${CUDA_ENABLE_LINEINFO}")
message(STATUS "CUPROJSHIM: Statically link the CUDA toolkit runtime and libraries: ${CUDA_STATIC_RUNTIME}")


rapids_cmake_build_type("Release")
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
include(../../../../cpp/cuproj/cmake/modules/ConfigureCUDA.cmake)

add_library(cuprojshim STATIC cuprojshim.cu)

set_target_properties(cuprojshim
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cuprojshim
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUPROJ_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUPROJ_CUDA_FLAGS}>"
)

target_compile_definitions(cuprojshim
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUPROJ_CXX_DEFINITIONS}>"
                   "$<$<COMPILE_LANGUAGE:CUDA>:${CUPROJ_CUDA_DEFINITIONS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(cuprojshim
           PUBLIC      "$<BUILD_INTERFACE:${CUPROJ_SOURCE_DIR}/include>"
           PRIVATE     "$<BUILD_INTERFACE:${CUPROJSHIM_SOURCE_DIR}>"
           INTERFACE   "$<INSTALL_INTERFACE:include>")

target_link_libraries(cuprojshim cuproj rmm::rmm cuspatial::cuspatial)

# Per-thread default stream
if(PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(cuprojshim PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

# Disable NVTX if necessary
if(NOT USE_NVTX)
    target_compile_definitions(cuprojshim PUBLIC NVTX_DISABLE)
endif()

# Define spdlog level
target_compile_definitions(cuprojshim PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")


add_library(cuproj::cuprojshim ALIAS cuprojshim)
