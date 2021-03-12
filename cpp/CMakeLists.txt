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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# If `CMAKE_CUDA_ARCHITECTURES` is not defined, build for all supported architectures. If
# `CMAKE_CUDA_ARCHITECTURES` is set to an empty string (""), build for only the current
# architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.

# This needs to be run before enabling the CUDA language due to the default initialization behavior
# of `CMAKE_CUDA_ARCHITECTURES`, https://gitlab.kitware.com/cmake/cmake/-/issues/21302
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CUSPATIAL_BUILD_FOR_ALL_ARCHS TRUE)
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
  set(CUSPATIAL_BUILD_FOR_DETECTED_ARCHS TRUE)
endif()

project(CUSPATIAL VERSION 0.19.0 LANGUAGES C CXX)

# Needed because GoogleBenchmark changes the state of FindThreads.cmake,
# causing subsequent runs to have different values for the `Threads::Threads` target.
# Setting this flag ensures `Threads::Threads` is the same value in first run and subsequent runs.
set(THREADS_PREFER_PTHREAD_FLAG ON)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" OFF)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
option(CUDF_USE_ARROW_STATIC "Build and statically link Arrow libraries" OFF)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

message(STATUS "CUSPATIAL: Build with NVTX support: ${USE_NVTX}")
message(STATUS "CUSPATIAL: Configure CMake to build tests: ${BUILD_TESTS}")
message(STATUS "CUSPATIAL: Configure CMake to build (google) benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "CUSPATIAL: Use a file cache for JIT compiled kernels: ${JITIFY_USE_CACHE}")
message(STATUS "CUSPATIAL: Build and statically link Arrow libraries: ${CUDF_USE_ARROW_STATIC}")
message(STATUS "CUSPATIAL: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(STATUS "CUSPATIAL: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNING}")
message(STATUS "CUSPATIAL: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler: ${CUDA_ENABLE_LINEINFO}")
message(STATUS "CUSPATIAL: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")
set(CUSPATIAL_BUILD_TESTS ${BUILD_TESTS})
set(CUSPATIAL_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})

set(CUSPATIAL_CXX_FLAGS "")
set(CUSPATIAL_CUDA_FLAGS "")
set(CUSPATIAL_CXX_DEFINITIONS "")
set(CUSPATIAL_CUDA_DEFINITIONS "")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "CUSPATIAL: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    message(STATUS "CUSPATIAL: Conda build detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "CUSPATIAL: Conda environment detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
include(cmake/Modules/ConfigureCUDA.cmake)

###################################################################################################
# - dependencies ----------------------------------------------------------------------------------

# find gdal
find_package(GDAL REQUIRED)
# add third party dependencies using CPM
include(cmake/thirdparty/CUSPATIAL_GetCPM.cmake)
# find or add cuDF
include(cmake/thirdparty/CUSPATIAL_GetCUDF.cmake)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cuspatial SHARED
    src/indexing/construction/point_quadtree.cu
    src/interpolate/cubic_spline.cu
    src/io/shp/polygon_shapefile_reader.cpp
    src/io/shp/polygon_shapefile_reader.cu
    src/join/quadtree_point_in_polygon.cu
    src/join/quadtree_point_to_nearest_polyline.cu
    src/join/quadtree_poly_filtering.cu
    src/spatial/polygon_bounding_box.cu
    src/spatial/polyline_bounding_box.cu
    src/spatial/point_in_polygon.cu
    src/spatial_window/spatial_window.cu
    src/spatial/haversine.cu
    src/spatial/hausdorff.cu
    src/spatial/lonlat_to_cartesian.cu
    src/trajectory/derive_trajectories.cu
    src/trajectory/trajectory_bounding_boxes.cu
    src/trajectory/trajectory_distances_and_speeds.cu
)

set_target_properties(cuspatial
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        14
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       14
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cuspatial
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_FLAGS}>"
)

target_compile_definitions(cuspatial
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_DEFINITIONS}>"
                   "$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_DEFINITIONS}>"
)

# Disable Jitify log printing. See https://github.com/NVIDIA/jitify/issues/79
target_compile_definitions(cuspatial PRIVATE "JITIFY_PRINT_LOG=0")

# Specify include paths for the current target and dependents
target_include_directories(cuspatial
           PUBLIC      "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}/include>"
           PRIVATE     "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}/src>"
           INTERFACE   "$<INSTALL_INTERFACE:include>")

# Add Conda library paths if specified
if(CONDA_LINK_DIRS)
    target_link_directories(cuspatial PUBLIC "$<BUILD_INTERFACE:${CONDA_LINK_DIRS}>")
endif()

# Add Conda include paths if specified
if(CONDA_INCLUDE_DIRS)
    target_include_directories(cuspatial PUBLIC "$<BUILD_INTERFACE:${CONDA_INCLUDE_DIRS}>")
endif()

# Per-thread default stream
if(PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(cuspatial PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

# Disable NVTX if necessary
if(NOT USE_NVTX)
    target_compile_definitions(cuspatial PUBLIC NVTX_DISABLE)
endif()

# Define spdlog level
target_compile_definitions(cuspatial PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

# Specify the target module library dependencies
target_link_libraries(cuspatial PUBLIC GDAL::GDAL cudf::cudf CUDA::cusparse)

add_library(cuspatial::cuspatial ALIAS cuspatial)

###################################################################################################
# - tests and benchmarks --------------------------------------------------------------------------
###################################################################################################
if (CUSPATIAL_BUILD_TESTS OR CUSPATIAL_BUILD_BENCHMARKS)
    # Find or install GoogleTest
    CPMFindPackage(NAME GTest
        VERSION         1.10.0
        GIT_REPOSITORY  https://github.com/google/googletest.git
        GIT_TAG         release-1.10.0
        GIT_SHALLOW     TRUE
        OPTIONS         "INSTALL_GTEST OFF"
        # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
        FIND_PACKAGE_ARGUMENTS "CONFIG")
    # Add GTest aliases if they don't already exist.
    # Assumes if GTest::gtest doesn't exist, the others don't either.
    # TODO: Is this always a valid assumption?
    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
        add_library(GTest::gmock ALIAS gmock)
        add_library(GTest::gtest_main ALIAS gtest_main)
        add_library(GTest::gmock_main ALIAS gmock_main)
    endif()
    if(GTest_ADDED)
        install(TARGETS gmock
                        gtest
                        gmock_main
                        gtest_main
            DESTINATION lib
            EXPORT cuspatial-targets)
    endif()
endif()

###################################################################################################
# - add tests -------------------------------------------------------------------------------------

if(CUSPATIAL_BUILD_TESTS)
    # include CTest module -- automatically calls enable_testing()
    include(CTest)
    add_subdirectory(tests)
endif()

###################################################################################################
# - add benchmarks --------------------------------------------------------------------------------

if(CUSPATIAL_BUILD_BENCHMARKS)
    # Find or install GoogleBench
    CPMFindPackage(NAME benchmark
        VERSION         1.5.2
        GIT_REPOSITORY  https://github.com/google/benchmark.git
        GIT_TAG         v1.5.2
        GIT_SHALLOW     TRUE
        OPTIONS         "BENCHMARK_ENABLE_TESTING OFF"
                        "BENCHMARK_ENABLE_INSTALL OFF")
    if(benchmark_ADDED)
        install(TARGETS benchmark
                        benchmark_main
            DESTINATION lib
            EXPORT cuspatial-targets)
    endif()
    add_subdirectory(benchmarks)
endif()

###################################################################################################
# - install targets -------------------------------------------------------------------------------

include(GNUInstallDirs)

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cuspatial)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cuspatial)

install(TARGETS cuspatial
        DESTINATION lib
        EXPORT cuspatial-targets)

install(DIRECTORY ${CUSPATIAL_SOURCE_DIR}/include/cuspatial
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/cuspatial-config.cmake.in "${CUSPATIAL_BINARY_DIR}/cmake/cuspatial-config.cmake"
                              INSTALL_DESTINATION "${INSTALL_CONFIGDIR}")

write_basic_package_version_file("${CUSPATIAL_BINARY_DIR}/cmake/cuspatial-config-version.cmake"
                                 COMPATIBILITY SameMinorVersion)

install(FILES   "${CUSPATIAL_BINARY_DIR}/cmake/cuspatial-config.cmake"
                "${CUSPATIAL_BINARY_DIR}/cmake/cuspatial-config-version.cmake"
    DESTINATION "${INSTALL_CONFIGDIR}")

install(EXPORT  cuspatial-targets
    FILE        cuspatial-targets.cmake
    NAMESPACE   cuspatial::
    DESTINATION "${INSTALL_CONFIGDIR}")

################################################################################################
# - build export -------------------------------------------------------------------------------

configure_package_config_file(cmake/cuspatial-build-config.cmake.in ${CUSPATIAL_BINARY_DIR}/cuspatial-config.cmake
                              INSTALL_DESTINATION ${CUSPATIAL_BINARY_DIR})

write_basic_package_version_file(${CUSPATIAL_BINARY_DIR}/cuspatial-config-version.cmake
                                 COMPATIBILITY SameMinorVersion)

export(EXPORT cuspatial-targets
    FILE ${CUSPATIAL_BINARY_DIR}/cuspatial-targets.cmake
    NAMESPACE   cuspatial::)
