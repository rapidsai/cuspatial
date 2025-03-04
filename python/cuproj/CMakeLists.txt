# =============================================================================
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(cuproj-python)

project(
  cuproj-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA)

option(FIND_CUPROJ_CPP "Search for existing cuproj C++ installations before defaulting to local files"
       OFF)

# If the user requested it we attempt to find cuproj.
if(FIND_CUPROJ_CPP)
  find_package(cuproj "${RAPIDS_VERSION}")
else()
  set(cuproj_FOUND OFF)
endif()

if(NOT cuproj_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(CUDA_STATIC_RUNTIME ON)

  add_subdirectory(../../cpp/cuproj cuproj-cpp EXCLUDE_FROM_ALL)

  # Since there are multiple subpackages of cuproj._lib that require access to libcuspatial, we place the
  # library in the cuproj directory as a single source of truth and modify the other rpaths
  # appropriately.
  set(cython_lib_dir cuproj)
  install(TARGETS cuproj DESTINATION ${cython_lib_dir})
endif()

add_subdirectory(cuproj/cuprojshim)

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cuproj/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cuproj PATHS "${cython_lib_dir}")
endif()
