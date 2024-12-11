#=============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

function(find_and_configure_cudf)

    if(TARGET cudf::cudf)
      return()
    endif()

    set(oneValueArgs VERSION GIT_REPO GIT_TAG USE_CUDF_STATIC EXCLUDE_FROM_ALL PER_THREAD_DEFAULT_STREAM)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(global_targets cudf::cudf)
    set(cudf_components "")

    if(BUILD_TESTS OR BUILD_BENCHMARKS)
      list(APPEND global_targets cudf::cudftestutil cudf::cudftestutil_impl)
      set(cudf_components COMPONENTS testing)
    endif()

    set(BUILD_SHARED ON)
    if(${PKG_USE_CUDF_STATIC})
        set(BUILD_SHARED OFF)
    endif()

    rapids_cpm_find(cudf ${PKG_VERSION} ${cudf_components}
      GLOBAL_TARGETS ${global_targets}
      BUILD_EXPORT_SET cuspatial-exports
      INSTALL_EXPORT_SET cuspatial-exports
      CPM_ARGS
        GIT_REPOSITORY   ${PKG_GIT_REPO}
        GIT_TAG          ${PKG_GIT_TAG}
        GIT_SHALLOW      TRUE
        SOURCE_SUBDIR    cpp
        EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS "BUILD_TESTS OFF"
                "BUILD_BENCHMARKS OFF"
                "BUILD_SHARED_LIBS ${BUILD_SHARED}"
                "CUDF_BUILD_TESTUTIL ${BUILD_TESTS}"
                "CUDF_BUILD_STREAMS_TEST_UTIL ${BUILD_TESTS}"
                "CUDF_USE_PER_THREAD_DEFAULT_STREAM ${PKG_PER_THREAD_DEFAULT_STREAM}"
    )

    if(TARGET cudf)
      set_property(TARGET cudf PROPERTY SYSTEM TRUE)
    endif()
endfunction()

set(CUSPATIAL_MIN_VERSION_cudf "${CUSPATIAL_VERSION_MAJOR}.${CUSPATIAL_VERSION_MINOR}")

if(NOT DEFINED CUSPATIAL_CUDF_GIT_REPO)
  set(CUSPATIAL_CUDF_GIT_REPO https://github.com/rapidsai/cudf.git)
endif()

if(NOT DEFINED CUSPATIAL_CUDF_GIT_TAG)
  set(CUSPATIAL_CUDF_GIT_TAG branch-${CUSPATIAL_MIN_VERSION_cudf})
endif()

find_and_configure_cudf(VERSION ${CUSPATIAL_MIN_VERSION_cudf}.00
                       GIT_REPO ${CUSPATIAL_CUDF_GIT_REPO}
                        GIT_TAG ${CUSPATIAL_CUDF_GIT_TAG}
                USE_CUDF_STATIC ${CUSPATIAL_USE_CUDF_STATIC}
               EXCLUDE_FROM_ALL ${CUSPATIAL_EXCLUDE_CUDF_FROM_ALL}
      PER_THREAD_DEFAULT_STREAM ${PER_THREAD_DEFAULT_STREAM})
