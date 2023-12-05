# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

set(cuspatial_version 24.02.00)

include(../../fetch_rapids.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(cuspatial-python)

project(
  cuspatial-python
  VERSION ${cuspatial_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA)

option(FIND_CUSPATIAL_CPP "Search for existing cuspatial C++ installations before defaulting to local files"
       OFF)

# When building the Python package, always defer to pyarrow to tell us where to find libarrow. This
# works whether libarrow is installed by pyarrow itself (i.e. as part of a wheel) or as part of a
# separate libarrow package (i.e. in conda) because pyarrow always knows where it is. libcudf must
# be built against an ABI-compatible arrow library for the package to load at runtime, so searching
# this way is always correct.
set(USE_LIBARROW_FROM_PYARROW ON)

# If the user requested it we attempt to find cuspatial.
if(FIND_CUSPATIAL_CPP)
  find_package(cuspatial ${cuspatial_version} REQUIRED)
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
  include(cmake/Modules/WheelHelpers.cmake)
  # TODO: This install is currently overzealous. We should only install the libraries that are
  # downloaded by CPM during the build, not libraries that were found on the system.  However, in
  # practice right this would only be a problem is if libcudf was not found but some of the
  # dependencies were, and we have no real use cases where that happens.
  install_aliased_imported_targets(
    TARGETS cuspatial arrow_shared nvcomp::nvcomp nvcomp::nvcomp_gdeflate nvcomp::nvcomp_bitcomp
    DESTINATION ${cython_lib_dir}
  )
endif()

include(rapids-cython)
rapids_cython_init()

add_subdirectory(cuspatial/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cuspatial PATHS "${cython_lib_dir}")
endif()
