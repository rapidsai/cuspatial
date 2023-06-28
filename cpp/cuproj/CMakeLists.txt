#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(../../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUPROJ)

project(CUPROJ VERSION 23.08.00 LANGUAGES C CXX CUDA)

# Needed because GoogleBenchmark changes the state of FindThreads.cmake,
# causing subsequent runs to have different values for the `Threads::Threads` target.
# Setting this flag ensures `Threads::Threads` is the same value in first run and subsequent runs.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Must come after enable_language(CUDA)
# Use `-isystem <path>` instead of `-isystem=<path>`
# because the former works with clangd intellisense
set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuproj shared libraries" ON)
option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" OFF)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)

message(STATUS "CUPROJ: Build with NVTX support: ${USE_NVTX}")
message(STATUS "CUPROJ: Configure CMake to build tests: ${BUILD_TESTS}")
message(STATUS "CUPROJ: Configure CMake to build (google) benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "CUPROJ: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(STATUS "CUPROJ: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNING}")
message(STATUS "CUPROJ: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler: ${CUDA_ENABLE_LINEINFO}")
message(STATUS "CUPROJ: Statically link the CUDA toolkit runtime and libraries: ${CUDA_STATIC_RUNTIME}")

# Set a default build type if none was specified
rapids_cmake_build_type("Release")
set(CUPROJ_BUILD_TESTS ${BUILD_TESTS})
set(CUPROJ_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})

set(CUPROJ_CXX_FLAGS "")
set(CUPROJ_CUDA_FLAGS "")
set(CUPROJ_CXX_DEFINITIONS "")
set(CUPROJ_CUDA_DEFINITIONS "")

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "CUPROJ: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

rapids_cmake_support_conda_env(conda_env)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
include(cmake/modules/ConfigureCUDA.cmake)

###################################################################################################
# - dependencies ----------------------------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()
# find or add cuDF
include(cmake/thirdparty/get_cudf.cmake)
# find or install GoogleTest and Proj
if (CUPROJ_BUILD_TESTS)
    include(cmake/thirdparty/get_gtest.cmake)
    include(cmake/thirdparty/get_proj.cmake)
endif()

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cuproj
    src/test.cu
)

set_target_properties(cuproj
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cuproj
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUPROJ_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUPROJ_CUDA_FLAGS}>"
)

target_compile_definitions(cuproj
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUPROJ_CXX_DEFINITIONS}>"
                   "$<$<COMPILE_LANGUAGE:CUDA>:${CUPROJ_CUDA_DEFINITIONS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(cuproj
           PUBLIC      "$<BUILD_INTERFACE:${CUPROJ_SOURCE_DIR}/include>"
           PRIVATE     "$<BUILD_INTERFACE:${CUPROJ_SOURCE_DIR}/src>"
           INTERFACE   "$<INSTALL_INTERFACE:include>")

# Add Conda library and include paths if specified
if(TARGET conda_env)
    target_link_libraries(cuproj PRIVATE "$<BUILD_INTERFACE:conda_env>")
endif()

# Per-thread default stream
if(PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(cuproj PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

# Disable NVTX if necessary
if(NOT USE_NVTX)
    target_compile_definitions(cuproj PUBLIC NVTX_DISABLE)
endif()

# Define spdlog level
target_compile_definitions(cuproj PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

# Specify the target module library dependencies
#target_link_libraries(cuproj PUBLIC cudf::cudf)

add_library(cuproj::cuproj ALIAS cuproj)

###################################################################################################
# - add tests -------------------------------------------------------------------------------------

if(CUPROJ_BUILD_TESTS)
    # include CTest module -- automatically calls enable_testing()
    include(CTest)
    add_subdirectory(tests)
endif()

###################################################################################################
# - add benchmarks --------------------------------------------------------------------------------

if(CUPROJ_BUILD_BENCHMARKS)
    # Find or install GoogleBench
    CPMFindPackage(NAME benchmark
        VERSION         1.5.3
        GIT_REPOSITORY  https://github.com/google/benchmark.git
        GIT_TAG         v1.5.3
        GIT_SHALLOW     TRUE
        OPTIONS         "BENCHMARK_ENABLE_TESTING OFF"
                        "BENCHMARK_ENABLE_INSTALL OFF")

    # Find or install NVBench Temporarily force downloading of fmt because current versions of nvbench
    # do not support the latest version of fmt, which is automatically pulled into our conda
    # environments by mamba.
    set(CPM_DOWNLOAD_fmt TRUE)
    include(${rapids-cmake-dir}/cpm/nvbench.cmake)
    rapids_cpm_nvbench()
    add_subdirectory(benchmarks)
endif()

###################################################################################################
# - install targets -------------------------------------------------------------------------------

rapids_cmake_install_lib_dir(lib_dir)
include(CPack)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cuproj)

install(TARGETS cuproj
        DESTINATION ${lib_dir}
        EXPORT cuproj-exports)

install(DIRECTORY ${CUPROJ_SOURCE_DIR}/include/cuproj
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(doc_string
    [=[
Provide targets for the cuproj library.

cuproj is a GPU-accelerated library for transformation of geospatial coordinates between coordinates
reference systems.

Imported Targets
^^^^^^^^^^^^^^^^

If cuproj is found, this module defines the following IMPORTED GLOBAL
targets:

 cuproj::cuproj             - The main cuproj library.
    ]=]
)

rapids_export(
  INSTALL cuproj
  EXPORT_SET cuproj-exports
  GLOBAL_TARGETS cuproj
  NAMESPACE cuproj::
  DOCUMENTATION doc_string
)


################################################################################################
# - build export -------------------------------------------------------------------------------

rapids_export(
  BUILD cuproj
  EXPORT_SET cuproj-exports
  GLOBAL_TARGETS cuproj
  NAMESPACE cuproj::
  DOCUMENTATION doc_string
)


# ##################################################################################################
# * build documentation ----------------------------------------------------------------------------

find_package(Doxygen)

if(DOXYGEN_FOUND)

# doc targets for cuproj
add_custom_command(
  OUTPUT CUPROJ_DOXYGEN
  WORKING_DIRECTORY ${CUPROJ_SOURCE_DIR}/doxygen
  COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
  VERBATIM
  COMMENT "Custom command for building cuproj doxygen docs."
)

add_custom_target(
  docs_cuproj
  DEPENDS CUPROJ_DOXYGEN
  COMMENT "Custom command for building cuproj doxygen docs."
)

endif()