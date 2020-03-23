#=============================================================================
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDA_SPATIAL VERSION 0.14.0 LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated-declarations")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "CUSPATIAL: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "CUSPATIAL: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# set warnings as errors
# TODO: remove `no-maybe-unitialized` used to suppress warnings in rmm::exec_policy
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror,-Wno-error=deprecated-declarations")

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
if (CMAKE_CUDA_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler -rdynamic")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

# To apply RUNPATH to transitive dependencies (this is a temporary solution)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

option(BUILD_TESTS "Configure CMake to build tests" ON)

option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{RMM_ROOT}/include")

find_library(RMM_LIBRARY "rmm"
             HINTS "$ENV{RMM_ROOT}/lib")

message(STATUS "RMM: RMM_ROOT set to $ENV{RMM_ROOT}")
message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if (RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif (RMM_INCLUDE AND RMM_LIBRARY)

# - CUDF -------------------------------------------------------------------------------------------

find_path(CUDF_INCLUDE "cudf"
          HINTS "$ENV{CUDF_ROOT}/include")

find_library(CUDF_LIBRARY "cudf"
             HINTS "$ENV{CUDF_ROOT}/lib")

SET(CUDF_SRC_INCLUDE "$ENV{CUDF_HOME}/cpp/src")
SET(CUB_INCLUDE "$ENV{CUDF_HOME}/cpp/thirdparty/cub")
SET(SIMT_INCLUDE "$ENV{CUDF_HOME}/cpp/thirdparty/libcudacxx/include")
SET(JITIFY_INCLUDE "$ENV{CUDF_HOME}/cpp/thirdparty/jitify")
SET(CUDF_TEST_INCLUDE "$ENV{CUDF_HOME}/cpp/")

message(STATUS "CUDF: CUDF_HOME set to $ENV{CUDF_HOME}")
message(STATUS "CUDF: CUDF_LIBRARY set to ${CUDF_LIBRARY}")
message(STATUS "CUDF: CUDF_INCLUDE set to ${CUDF_INCLUDE}")
message(STATUS "CUDF SRC: CUDF_SRC_INCLUDE set to ${CUDF_SRC_INCLUDE}")
message(STATUS "CUDF: CUB_INCLUDE set to ${CUB_INCLUDE}")
message(STATUS "CUDF: CUB_INCLUDE set to ${SIMT_INCLUDE}")
message(STATUS "CUDF: SIMT_INCLUDE set to ${JITIFY_INCLUDE}")
message(STATUS "CUDF: CUDF_TEST_INCLUDE set to ${CUDF_TEST_INCLUDE}")

add_library(cudf SHARED IMPORTED ${CUDF_LIBRARY})
if (CUDF_INCLUDE AND CUDF_LIBRARY)
    set_target_properties(cudf PROPERTIES IMPORTED_LOCATION ${CUDF_LIBRARY})
endif (CUDF_INCLUDE AND CUDF_LIBRARY)

###################################################################################################
# - NVStrings -------------------------------------------------------------------------------------

find_path(NVSTRINGS_INCLUDE "nvstrings"
          HINTS "$ENV{NVSTRINGS_ROOT}/include")

find_library(NVSTRINGS_LIBRARY "NVStrings"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib")

find_library(NVCATEGORY_LIBRARY "NVCategory"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib")

find_library(NVTEXT_LIBRARY "NVText"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib")

message(STATUS "NVSTRINGS: NVSTRINGS_INCLUDE set to ${NVSTRINGS_INCLUDE}")
message(STATUS "NVSTRINGS: NVSTRINGS_LIBRARY set to ${NVSTRINGS_LIBRARY}")
message(STATUS "NVSTRINGS: NVCATEGORY_LIBRARY set to ${NVCATEGORY_LIBRARY}")
message(STATUS "NVSTRINGS: NVTEXT_LIBRARY set to ${NVTEXT_LIBRARY}")

add_library(NVStrings SHARED IMPORTED ${NVSTRINGS_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)
    set_target_properties(NVStrings PROPERTIES IMPORTED_LOCATION ${NVSTRINGS_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)

add_library(NVCategory SHARED IMPORTED ${NVCATEGORY_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVCATEGORY_LIBRARY)
    set_target_properties(NVCategory PROPERTIES IMPORTED_LOCATION ${NVCATEGORY_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVCATEGORY_LIBRARY)

add_library(NVText SHARED IMPORTED ${NVTEXT_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVTEXT_LIBRARY)
    set_target_properties(NVText PROPERTIES IMPORTED_LOCATION ${NVTEXT_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVTEXT_LIBRARY)


# - find gdal -------------------------------------------------------------------------------------

find_package(GDAL REQUIRED)

message(STATUS "GDAL: GDAL_LIBRARIES set to ${GDAL_LIBRARIES}")
message(STATUS "GDAL: GDAL_INCLUDE_DIRS set to ${GDAL_INCLUDE_DIRS}")

if(NOT GDAL_FOUND)
    message(FATAL_ERROR "GDAL not found, please check your settings.")
endif(NOT GDAL_FOUND)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories(${GTEST_INCLUDE_DIR})
        add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)

  include(ConfigureGoogleBenchmark)

  if(GBENCH_FOUND)
    message(STATUS "Google C++ Benchmarking Framework (Google Benchmark) found in ${GBENCH_ROOT}")
    include_directories(${GBENCH_INCLUDE_DIR})
    add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks)
  else()
    message(AUTHOR_WARNING "Google C++ Benchmarking Framework (Google Benchmark) not found: automated tests are disabled.")
  endif(GBENCH_FOUND)

endif(BUILD_BENCHMARKS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/../thirdparty/cub"
                    "${GDAL_INCLUDE_DIR}"
                    "${RMM_INCLUDE}"
                    "${CUDF_SRC_INCLUDE}"
                    "${CUDF_INCLUDE}"
                    "${CUB_INCLUDE}"
                    "${SIMT_INCLUDE}"
                    "${JITIFY_INCLUDE}")


###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GDAL_LIBRARIES}"
                 "${CONDA_LINK_DIRS}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}"
                 "${CUDF_LIBRARY}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cuspatial SHARED
	    src/join/pip_refine.cu
	    src/join/quadtree_polygon.cu
	    src/spatial/bounding_box.cu
	    src/indexing/quadtree.cu
            src/io/shp/polygon_shapefile_reader.cu
            src/io/soa/polygon_soa_reader.cu
            src/io/soa/point_soa_reader.cu
            src/io/soa/uint32_soa_reader.cu
            src/io/soa/timestamp_soa_reader.cu
            src/query/spatial_window_points.cu
            src/spatial/point_in_polygon.cu
            src/spatial/haversine.cu
            src/spatial/hausdorff.cu
            src/spatial/lonlat_to_coordinate.cu
            src/trajectory/trajectory_spatial_bound.cu
            src/trajectory/trajectory_distance_speed.cu
            src/trajectory/derive_trajectories.cu
            src/trajectory/subset_trajectories.cu
            src/utility/utility.cpp)

#Override RPATH for cuspatial
SET_TARGET_PROPERTIES(cuspatial PROPERTIES BUILD_RPATH "\$ORIGIN")

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(cuspatial ${NVTX_LIBRARY})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX")
endif(USE_NVTX)


###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(cuspatial cudf rmm cudart cuda nvrtc gdal)


###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS cuspatial
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cuspatial
        DESTINATION include)
