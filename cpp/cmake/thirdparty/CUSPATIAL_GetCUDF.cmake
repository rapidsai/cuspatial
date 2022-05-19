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

function(find_and_configure_cudf VERSION)

    if(TARGET cudf::cudf)
        return()
    endif()

    if(${VERSION} MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=])
        set(MAJOR_AND_MINOR "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    else()
        set(MAJOR_AND_MINOR "${VERSION}")
    endif()

    set(global_targets cudf::cudf)
    set(find_package_args "")
    if(BUILD_TESTS)
      list(APPEND global_targets cudf::cudftestutil)
      set(find_package_args "COMPONENTS testing")
    endif()

    rapids_cpm_find(
      cudf ${VERSION}
      GLOBAL_TARGETS "${global_targets}"
      BUILD_EXPORT_SET cuspatial-exports
      INSTALL_EXPORT_SET cuspatial-exports
      CPM_ARGS
      GIT_REPOSITORY https://github.com/rapidsai/cudf.git
      GIT_TAG branch-${MAJOR_AND_MINOR}
      GIT_SHALLOW TRUE
      OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF"
      FIND_PACKAGE_ARGUMENTS "${find_package_args}"
    )
endfunction()

set(CUSPATIAL_MIN_VERSION_cudf "${CUSPATIAL_VERSION_MAJOR}.${CUSPATIAL_VERSION_MINOR}.00")

find_and_configure_cudf(${CUSPATIAL_MIN_VERSION_cudf})
