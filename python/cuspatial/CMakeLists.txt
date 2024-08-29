# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

include(../../rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(cuspatial-python)

project(
  cuspatial-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA)

option(FIND_CUSPATIAL_CPP "Search for existing cuspatial C++ installations before defaulting to local files"
       OFF)

# If the user requested it we attempt to find cuspatial.
if(FIND_CUSPATIAL_CPP)
  find_package(cuspatial "${RAPIDS_VERSION}")
else()
  set(cuspatial_FOUND OFF)
endif()

if(NOT cuspatial_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(CUDA_STATIC_RUNTIME ON)
  set(CUSPATIAL_USE_CUDF_STATIC ON)
  set(CUSPATIAL_EXCLUDE_CUDF_FROM_ALL ON)

  add_subdirectory(../../cpp cuspatial-cpp EXCLUDE_FROM_ALL)

  set(cython_lib_dir cuspatial)
  install(TARGETS cuspatial DESTINATION ${cython_lib_dir})
endif()

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cuspatial/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cuspatial PATHS "${cython_lib_dir}")
endif()
