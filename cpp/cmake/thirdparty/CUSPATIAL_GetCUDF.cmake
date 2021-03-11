#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

function(cuspatial_save_if_enabled var)
    if(CUSPATIAL_${var})
        unset(${var} PARENT_SCOPE)
        unset(${var} CACHE)
    endif()
endfunction()

function(cuspatial_restore_if_enabled var)
    if(CUSPATIAL_${var})
        set(${var} ON CACHE INTERNAL "" FORCE)
    endif()
endfunction()

function(find_and_configure_cudf VERSION)
    cuspatial_save_if_enabled(BUILD_TESTS)
    cuspatial_save_if_enabled(BUILD_BENCHMARKS)
    CPMFindPackage(NAME cudf
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/trxcllnt/cudf.git
        # GIT_TAG         branch-${VERSION}
        GIT_TAG         fix/cmake-always-export-cudftestutil
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS NO"
                        "BUILD_BENCHMARKS NO"
                        "USE_NVTX ${USE_NVTX}"
                        "JITIFY_USE_CACHE ${JITIFY_USE_CACHE}"
                        "CUDA_STATIC_RUNTIME ${CUDA_STATIC_RUNTIME}"
                        "CUDF_USE_ARROW_STATIC ${CUDF_USE_ARROW_STATIC}"
                        "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}"
                        "PER_THREAD_DEFAULT_STREAM ${PER_THREAD_DEFAULT_STREAM}"
                        "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNING}")
    cuspatial_restore_if_enabled(BUILD_TESTS)
    cuspatial_restore_if_enabled(BUILD_BENCHMARKS)

    # Make sure consumers of cuspatial can also see cudf::cudf
    if(TARGET cudf::cudf)
        get_target_property(cudf_is_imported cudf::cudf IMPORTED)
        if(cudf_is_imported)
            set_target_properties(cudf::cudf PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
        get_target_property(cudftestutil_is_imported cudf::cudftestutil IMPORTED)
        if(cudftestutil_is_imported)
            set_target_properties(cudf::cudftestutil PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
    endif()

    if(NOT cudf_BINARY_DIR IN_LIST CMAKE_PREFIX_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${cudf_BINARY_DIR}")
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
    endif()
endfunction()

set(CUSPATIAL_MIN_VERSION_cudf "${CUSPATIAL_VERSION_MAJOR}.${CUSPATIAL_VERSION_MINOR}")

find_and_configure_cudf(${CUSPATIAL_MIN_VERSION_cudf})
