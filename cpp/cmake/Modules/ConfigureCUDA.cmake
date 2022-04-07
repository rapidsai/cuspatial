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

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUSPATIAL_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations)
    if(CUSPATIAL_BUILD_TESTS OR CUSPATIAL_BUILD_BENCHMARKS)
        # Suppress parentheses warning which causes gmock to fail
        list(APPEND CUSPATIAL_CUDA_FLAGS -Xcompiler=-Wno-parentheses)
    endif()
endif(CMAKE_COMPILER_IS_GNUCXX)

list(APPEND CUSPATIAL_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2.0)
    list(APPEND CUSPATIAL_CUDA_FLAGS -Werror=all-warnings)
endif()
list(APPEND CUSPATIAL_CUDA_FLAGS -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations)

# Produce smallest binary size
list(APPEND CUSPATIAL_CUDA_FLAGS -Xfatbin=-compress-all)

if(DISABLE_DEPRECATION_WARNING)
    list(APPEND CUSPATIAL_CXX_FLAGS -Wno-deprecated-declarations)
    list(APPEND CUSPATIAL_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
if(CUDA_ENABLE_LINEINFO)
    list(APPEND CUSPATIAL_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "CUSPATIAL: Building with debugging flags")
    list(APPEND CUSPATIAL_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()
