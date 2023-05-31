#=============================================================================
# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

include(../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUSPATIAL)

project(CUSPATIAL VERSION 23.06.00 LANGUAGES C CXX CUDA)

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

option(BUILD_SHARED_LIBS "Build cuSpatial shared libraries" ON)
option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" OFF)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)

message(STATUS "CUSPATIAL: Build with NVTX support: ${USE_NVTX}")
message(STATUS "CUSPATIAL: Configure CMake to build tests: ${BUILD_TESTS}")
message(STATUS "CUSPATIAL: Configure CMake to build (google) benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "CUSPATIAL: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(STATUS "CUSPATIAL: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNING}")
message(STATUS "CUSPATIAL: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler: ${CUDA_ENABLE_LINEINFO}")
message(STATUS "CUSPATIAL: Statically link the CUDA toolkit runtime and libraries: ${CUDA_STATIC_RUNTIME}")

# Set a default build type if none was specified
rapids_cmake_build_type("Release")
set(CUSPATIAL_BUILD_TESTS ${BUILD_TESTS})
set(CUSPATIAL_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})

set(CUSPATIAL_CXX_FLAGS "")
set(CUSPATIAL_CUDA_FLAGS "")
set(CUSPATIAL_CXX_DEFINITIONS "")
set(CUSPATIAL_CUDA_DEFINITIONS "")

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "CUSPATIAL: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------
set(_ctk_static_suffix "")
if(CUDA_STATIC_RUNTIME)
  set(_ctk_static_suffix "_static")
  # Control legacy FindCUDA.cmake behavior too
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
endif()

rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET cuspatial-exports
  INSTALL_EXPORT_SET cuspatial-exports
)
# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
include(cmake/Modules/ConfigureCUDA.cmake)

###################################################################################################
# - dependencies ----------------------------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()
# find or add cuDF
include(cmake/thirdparty/CUSPATIAL_GetCUDF.cmake)

if (USE_NVTX)

rapids_cpm_find(
    NVTX3 3.1.0
    GLOBAL_TARGETS nvtx3-cpp
    CPM_ARGS
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.0-c-cpp
        GIT_SHALLOW TRUE
        DOWNLOAD_ONLY TRUE
)
include(${NVTX3_SOURCE_DIR}/nvtxImportedTargets.cmake)

endif()

# find or install GoogleTest
if (CUSPATIAL_BUILD_TESTS)
    include(cmake/thirdparty/get_gtest.cmake)
endif()

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cuspatial
    src/bounding_boxes/linestring_bounding_boxes.cu
    src/bounding_boxes/polygon_bounding_boxes.cu
    src/column/geometry_column_view.cpp
    src/distance/hausdorff.cu
    src/distance/haversine.cu
    src/distance/linestring_distance.cu
    src/distance/linestring_polygon_distance.cu
    src/distance/point_distance.cu
    src/distance/point_linestring_distance.cu
    src/distance/point_polygon_distance.cu
    src/distance/polygon_distance.cu
    src/equality/pairwise_multipoint_equals_count.cu
    src/indexing/point_quadtree.cu
    src/intersection/linestring_intersection.cu
    src/join/quadtree_point_in_polygon.cu
    src/join/quadtree_point_to_nearest_linestring.cu
    src/join/quadtree_bbox_filtering.cu
    src/nearest_points/point_linestring_nearest_points.cu
    src/point_in_polygon/point_in_polygon.cu
    src/points_in_range/points_in_range.cu
    src/projection/sinusoidal_projection.cu
    src/trajectory/derive_trajectories.cu
    src/trajectory/trajectory_bounding_boxes.cu
    src/trajectory/trajectory_distances_and_speeds.cu
)

set_target_properties(cuspatial
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

target_compile_options(cuspatial
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_FLAGS}>"
)

target_compile_definitions(cuspatial
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_DEFINITIONS}>"
                   "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_DEFINITIONS}>>"
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
target_link_libraries(cuspatial PUBLIC cudf::cudf)
target_link_libraries(cuspatial PRIVATE nvtx3-cpp)

add_library(cuspatial::cuspatial ALIAS cuspatial)

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
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cuspatial)

install(TARGETS cuspatial
        DESTINATION ${lib_dir}
        EXPORT cuspatial-exports)

install(DIRECTORY ${CUSPATIAL_SOURCE_DIR}/include/cuspatial
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message("Installing ${NVTX3_SOURCE_DIR}/include/nvtx3 to /home/coder/scratch")

install(DIRECTORY ${NVTX3_SOURCE_DIR}
        DESTINATION /home/coder/scratch)

set(doc_string
    [=[
Provide targets for the cuSpatial library.

cuSpatial is a GPU-accelerated library for spatial data management and analytics.

Imported Targets
^^^^^^^^^^^^^^^^

If cuspatial is found, this module defines the following IMPORTED GLOBAL
targets:

 cuspatial::cuspatial             - The main cuspatial library.
    ]=]
)

rapids_export(
  INSTALL cuspatial
  EXPORT_SET cuspatial-exports
  GLOBAL_TARGETS cuspatial
  NAMESPACE cuspatial::
  DOCUMENTATION doc_string
)

set(nvtx3_final_code_block
    [=[
      if(NOT TARGET nvtx3-c)
        add_library(nvtx3-c INTERFACE IMPORTED GLOBAL)
        set_target_properties(nvtx3-c PROPERTIES VERSION ${NVTX3_VERSION})
        target_include_directories(nvtx3-c INTERFACE
        set_target_properties(nvtx3-c PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include")
        target_link_libraries(nvtx3-c INTERFACE ${CMAKE_DL_LIBS})

        #-------------------------------------------------------
        # Define "nvtx3-cpp" library for the NVTX v3 C++ API
        # Separate target allows attaching independent compiler requirements if needed
        add_library(nvtx3-cpp INTERFACE IMPORTED GLOBAL)
        set_target_properties(nvtx3-cpp PROPERTIES VERSION ${NVTX3_VERSION})
        target_link_libraries(nvtx3-cpp INTERFACE nvtx3-c)
      endif()
    ]=]
)

rapids_export(
  INSTALL nvtx3-cpp
  EXPORT_SET cuspatial-exports
  GLOBAL_TARGETS nvtx3-cpp
  FINAL_CODE_BLOCK nvtx3_final_code_block
)

################################################################################################
# - build export -------------------------------------------------------------------------------

rapids_export(
  BUILD cuspatial
  EXPORT_SET cuspatial-exports
  GLOBAL_TARGETS cuspatial
  NAMESPACE cuspatial::
  DOCUMENTATION doc_string
)

rapids_export(
  BUILD nvtx3-cpp
  EXPORT_SET cuspatial-exports
  GLOBAL_TARGETS nvtx3-cpp
  FINAL_CODE_BLOCK nvtx3_final_code_block
)

# ##################################################################################################
# * build documentation ----------------------------------------------------------------------------

find_package(Doxygen)

if(DOXYGEN_FOUND)

# doc targets for cuSpatial
add_custom_command(
  OUTPUT CUSPATIAL_DOXYGEN
  WORKING_DIRECTORY ${CUSPATIAL_SOURCE_DIR}/doxygen
  COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
  VERBATIM
  COMMENT "Custom command for building cuspatial doxygen docs."
)

add_custom_target(
  docs_cuspatial
  DEPENDS CUSPATIAL_DOXYGEN
  COMMENT "Custom command for building cuspatial doxygen docs."
)

endif()
