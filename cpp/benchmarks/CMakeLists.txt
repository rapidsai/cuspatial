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

# Find or install GoogleTest
CPMFindPackage(NAME GTest
    VERSION         1.10.0
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         release-1.10.0
    GIT_SHALLOW     TRUE
    OPTIONS         "INSTALL_GTEST OFF"
    # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
    FIND_PACKAGE_ARGUMENTS "CONFIG")

if(GTest_ADDED)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gmock ALIAS gmock)
    add_library(GTest::gtest_main ALIAS gtest_main)
    add_library(GTest::gmock_main ALIAS gmock_main)
endif()

# Find or install GoogleBench
CPMFindPackage(NAME benchmark
    VERSION         1.5.2
    GIT_REPOSITORY  https://github.com/google/benchmark.git
    GIT_TAG         v1.5.2
    GIT_SHALLOW     TRUE
    OPTIONS         "BENCHMARK_ENABLE_TESTING OFF"
                    "BENCHMARK_ENABLE_INSTALL OFF")

# # find cudftestutil library
# find_library(CUDFTESTUTIL_LIBRARY "libcudftestutil.a"
#              HINTS "$ENV{CUDF_ROOT}/lib")

# message(STATUS "CUSPATIAL: CUDFTESTUTIL_LIBRARY set to ${CUDFTESTUTIL_LIBRARY}")

# add_library(cudftestutil STATIC IMPORTED ${CUDFTESTUTIL_LIBRARY})

# set_target_properties(cudftestutil
#     PROPERTIES CXX_STANDARD              14
#                CXX_STANDARD_REQUIRED     ON
#                CUDA_STANDARD             14
#                CUDA_STANDARD_REQUIRED    ON
#                IMPORTED_LOCATION         ${CUDFTESTUTIL_LIBRARY}
# )

# target_link_libraries(cudftestutil
#                PUBLIC GTest::gmock
#                       GTest::gtest
#                       GTest::gmock_main
#                       GTest::gtest_main
#                       pthread)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC}
                   "${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp")
    target_compile_options(${CMAKE_BENCH_NAME}
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_FLAGS}>")
    target_include_directories(${CMAKE_BENCH_NAME}
                PRIVATE "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}>"
                        "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}/src>")
    set_target_properties(${CMAKE_BENCH_NAME}
        PROPERTIES CXX_STANDARD              14
                   CXX_STANDARD_REQUIRED     ON
                   CUDA_STANDARD             14
                   CUDA_STANDARD_REQUIRED    ON
                   POSITION_INDEPENDENT_CODE ON
                   RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUSPATIAL_BINARY_DIR}/gbenchmarks>")
    target_link_libraries(${CMAKE_BENCH_NAME} cudftestutil cuspatial)
endfunction(ConfigureBench)

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - hausdorff benchmark ---------------------------------------------------------------------------

set(HAUSDORFF_BENCH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/hausdorff_benchmark.cpp")

ConfigureBench(HAUSDORFF_BENCH "${HAUSDORFF_BENCH_SRC}")
