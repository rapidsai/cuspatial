# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

project(
  libcuspatial-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

# Check if cuspatial is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuspatial package.
find_package(cuspatial "${RAPIDS_VERSION}")
if(cuspatial_FOUND)
  return()
endif()

unset(cuspatial_FOUND)

# # Find Python early so that later commands can use it
# find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)

# set options that only have 1 correct values in the context of libcuspatial wheel builds
set(BUILD_SHARED_LIBS ON)
set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUDA_ENABLE_LINEINFO OFF)
set(CUDA_STATIC_RUNTIME OFF)
set(CUSPATIAL_USE_CUDF_STATIC OFF)
set(CUSPATIAL_EXCLUDE_CUDF_FROM_ALL ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(../../cpp cuspatial-cpp)
install(TARGETS cuspatial DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
