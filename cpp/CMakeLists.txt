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

project(CUSPATIAL VERSION 0.19.0 LANGUAGES C CXX)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" OFF)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
option(ARROW_STATIC_LIB "Build and statically link Arrow libraries" OFF)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(AUTO_DETECT_CUDA_ARCHITECTURES "Auto-detect CUDA architectures to build from the GPUs on the current system" OFF)
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

message(STATUS "CUSPATIAL: Build with NVTX support: ${USE_NVTX}")
message(STATUS "CUSPATIAL: Configure CMake to build tests: ${BUILD_TESTS}")
message(STATUS "CUSPATIAL: Configure CMake to build (google) benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "CUSPATIAL: Use a file cache for JIT compiled kernels: ${JITIFY_USE_CACHE}")
message(STATUS "CUSPATIAL: Build and statically link Arrow libraries: ${ARROW_STATIC_LIB}")
message(STATUS "CUSPATIAL: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(STATUS "CUSPATIAL: Auto-detect CUDA architectures to build from the GPUs on the current system: ${AUTO_DETECT_CUDA_ARCHITECTURES}")
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
# find cuDF
include(cmake/thirdparty/CUSPATIAL_GetCUDF.cmake)

###################################################################################################
# - add tests -------------------------------------------------------------------------------------

if(CUSPATIAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

###################################################################################################
# - add benchmarks --------------------------------------------------------------------------------

if(CUSPATIAL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()


###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cuspatial SHARED
    "${CUSPATIAL_SOURCE_DIR}/src/indexing/construction/point_quadtree.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/interpolate/cubic_spline.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/io/shp/polygon_shapefile_reader.cpp"
    "${CUSPATIAL_SOURCE_DIR}/src/io/shp/polygon_shapefile_reader.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/join/quadtree_point_in_polygon.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/join/quadtree_point_to_nearest_polyline.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/join/quadtree_poly_filtering.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/spatial/polygon_bounding_box.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/spatial/polyline_bounding_box.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/spatial/point_in_polygon.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/spatial_window/spatial_window.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/spatial/haversine.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/spatial/hausdorff.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/spatial/lonlat_to_cartesian.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/trajectory/derive_trajectories.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/trajectory/trajectory_bounding_boxes.cu"
    "${CUSPATIAL_SOURCE_DIR}/src/trajectory/trajectory_distances_and_speeds.cu"
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

# Instruct jitify to use the kernel JIT cache
if(JITIFY_USE_CACHE)
    target_compile_definitions(cuspatial PUBLIC JITIFY_USE_CACHE "CUDF_VERSION=${CMAKE_PROJECT_VERSION}")
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
target_link_libraries(cuspatial PUBLIC GDAL::GDAL rmm::rmm cudf::cudf)

if(CUDA_STATIC_RUNTIME)
    target_link_libraries(cuspatial PUBLIC CUDA::nvrtc CUDA::cudart_static CUDA::cuda_driver)
else()
    target_link_libraries(cuspatial PUBLIC CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
endif()

add_library(cuspatial::cuspatial ALIAS cuspatial)

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

configure_package_config_file(cmake/cuspatial-config.cmake.in "${CUSPATIAL_BINARY_DIR}/cuspatial-config.cmake"
                              INSTALL_DESTINATION "${INSTALL_CONFIGDIR}")

write_basic_package_version_file("${CUSPATIAL_BINARY_DIR}/cuspatial-config-version.cmake"
                                 COMPATIBILITY SameMinorVersion)

install(FILES   "${CUSPATIAL_BINARY_DIR}/cuspatial-config.cmake"
                "${CUSPATIAL_BINARY_DIR}/cuspatial-config-version.cmake"
    DESTINATION "${INSTALL_CONFIGDIR}")

install(EXPORT  cuspatial-targets
    FILE        cuspatial-targets.cmake
    NAMESPACE   cuspatial::
    DESTINATION "${INSTALL_CONFIGDIR}")
