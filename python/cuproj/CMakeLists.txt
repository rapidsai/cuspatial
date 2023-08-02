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

include(../../fetch_rapids.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(cuproj-python)

project(
  cuproj-python
  VERSION ${cuproj_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA)

option(FIND_CUPROJ_CPP "Search for existing cuproj C++ installations before defaulting to local files"
       OFF)

option(CUPROJ_BUILD_WHEELS "Whether this build is generating a Python wheel." OFF)

# If the user requested it we attempt to find cuproj.
if(FIND_CUPROJ_CPP)
  find_package(cuproj ${cuproj_version})
else()
  set(cuproj_FOUND OFF)
endif()

if(NOT cuproj_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(_exclude_from_all "")
  if(CUPROJ_BUILD_WHEELS)

    # Statically link cudart if building wheels
    set(CUDA_STATIC_RUNTIME ON)

    # Need to set this so all the cuproj targets are global, not only cuproj::cuproj
    # https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_PACKAGE_TARGETS_GLOBAL.html#variable:CMAKE_FIND_PACKAGE_TARGETS_GLOBAL
    set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

    # Don't install the cuProj C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../../cpp/cuproj cuproj-cpp ${_exclude_from_all})

  set(cython_lib_dir cuproj)

  if(CUPROJ_BUILD_WHEELS)
    include(../cuspatial/cmake/Modules/WheelHelpers.cmake)
  endif()

  # Since there are multiple subpackages of cuproj._lib that require access to libcuspatial, we place the
  # library in the cuproj directory as a single source of truth and modify the other rpaths
  # appropriately.
  install(TARGETS cuproj DESTINATION ${cython_lib_dir})
endif()

add_subdirectory(cuproj/cuprojshim)

include(rapids-cython)
rapids_cython_init()

add_subdirectory(cuproj/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cuproj PATHS "${cython_lib_dir}")
endif()
