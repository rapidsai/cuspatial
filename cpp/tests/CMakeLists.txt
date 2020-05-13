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
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUSPATIAL_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

find_library(CUDFTESTUTIL_LIBRARY "libcudftestutil.a"
             HINTS "$ENV{CUDF_ROOT}/lib")

message(STATUS "CUDFTESTUTIL: CUDFTESTUTIL_LIBRARY set to ${CUDFTESTUTIL_LIBRARY}")
message(STATUS "CUDFTESTUTIL: CUDF_TEST_INCLUDE set to ${CUDF_TEST_INCLUDE}")

add_library(cudftestutil STATIC IMPORTED ${CUDFTESTUTIL_LIBRARY})

if (CUDF_TEST_INCLUDE AND CUDFTESTUTIL_LIBRARY)
    set_target_properties(cudftestutil PROPERTIES IMPORTED_LOCATION ${CUDFTESTUTIL_LIBRARY})
endif (CUDF_TEST_INCLUDE AND CUDFTESTUTIL_LIBRARY)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

option(BUILD_LEGACY_TESTS "Build cuSpatial legacy tests" ON)

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    string(REGEX MATCH "LEGACY_(.*)?_TEST" is_legacy_test "${CMAKE_TEST_NAME}")
    if(BUILD_LEGACY_TESTS OR (NOT is_legacy_test))
        add_executable(${CMAKE_TEST_NAME}
                    ${CMAKE_TEST_SRC})
        set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cuspatial cudf
                            cudftestutil rmm cudart cuda "${ARROW_LIB}" ${ZLIB_LIBRARIES} NVCategory
                            NVStrings nvrtc gdal)
        if(USE_NVTX)
            target_link_libraries(${CMAKE_TEST_NAME} ${NVTX_LIBRARY})
        endif(USE_NVTX)
        set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
        add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    endif(BUILD_LEGACY_TESTS OR (NOT is_legacy_test))
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/../thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/../thirdparty/libcudacxx/include"
                    "${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    "${GTEST_INCLUDE_DIR}"
                    "${RMM_INCLUDE}"
                    "${CUDF_INCLUDE}"
                    "${CUDF_SRC_INCLUDE}"
                    "${CUDF_TEST_INCLUDE}"
                    "${CUB_INCLUDE}"
                    )
 
###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GDAL_LIBRARIES}"
                 "${CONDA_LINK_DIRS}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}"
                 "${CUDF_LIBRARY}"
                 "${CUSPATIAL_LIBRARY}"
                 )
set(CUBIC_SPLINE_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/interpolate/cubic_spline_test.cpp")
 ConfigureTest(CUBIC_SPLINE_TEST "${CUBIC_SPLINE_TEST_SRC}")

set(LEGACY_COORDINATE_TRANSFORM_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/coordinate_transform_test.cu")
ConfigureTest(LEGACY_COORDINATE_TRANSFORM_TEST "${LEGACY_COORDINATE_TRANSFORM_TEST_SRC}")

set(COORDINATE_TRANSFORM_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/coordinate_transform_test.cu")
ConfigureTest(COORDINATE_TRANSFORM_TEST "${COORDINATE_TRANSFORM_TEST_SRC}")

set(LEGACY_HAVERSINE_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/haversine_toy.cu")
ConfigureTest(LEGACY_HAVERSINE_TEST "${LEGACY_HAVERSINE_TEST_SRC}")

set(LEGACY_HAUSDORFF_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/hausdorff_toy.cu"
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/hausdorff_test.cu"
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/hausdorff_compare.cu")
ConfigureTest(LEGACY_HAUSDORFF_TEST "${LEGACY_HAUSDORFF_TEST_SRC}")

set(HAUSDORFF_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/hausdorff_test.cpp")
ConfigureTest(HAUSDORFF_TEST "${HAUSDORFF_TEST_SRC}")

set(LEGACY_POINT_IN_POLYGON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/pip_toy.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/pip_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/pip_compare.cu")
ConfigureTest(LEGACY_POINT_IN_POLYGON_TEST "${LEGACY_POINT_IN_POLYGON_TEST_SRC}")

set(POINT_IN_POLYGON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/point_in_polygon_test.cpp")
ConfigureTest(POINT_IN_POLYGON_TEST "${POINT_IN_POLYGON_TEST_SRC}")

set(LEGACY_SHAPEFILE_POLYGON_READER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/read_shapefile_polygon_test.cu")
ConfigureTest(LEGACY_SHAPEFILE_POLYGON_READER_TEST "${LEGACY_SHAPEFILE_POLYGON_READER_TEST_SRC}")

set(LEGACY_SPATIAL_WINDOW_POINT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial_window/legacy/spatial_window_test_toy.cu")
ConfigureTest(LEGACY_SPATIAL_WINDOW_POINT_TEST "${LEGACY_SPATIAL_WINDOW_POINT_TEST_SRC}")

set(LEGACY_TRAJECTORY_DISTANCE_SPEED_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/trajectory/legacy/test_trajectory_distance_speed.cu")
ConfigureTest(LEGACY_TRAJECTORY_DISTANCE_SPEED_TEST "${LEGACY_TRAJECTORY_DISTANCE_SPEED_TEST_SRC}")

set(LEGACY_TRAJECTORY_DERIVE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/trajectory/legacy/test_trajectory_derive.cu")
ConfigureTest(LEGACY_TRAJECTORY_DERIVE_TEST "${LEGACY_TRAJECTORY_DERIVE_TEST_SRC}")

set(LEGACY_TRAJECTORY_SPATIAL_BOUND_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/trajectory/legacy/trajectory_spatial_bounds_toy.cu")
ConfigureTest(LEGACY_TRAJECTORY_SPATIAL_BOUND_TEST "${LEGACY_TRAJECTORY_SPATIAL_BOUND_SRC}")

set(LEGACY_TRAJECTORY_SUBSET_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/trajectory/legacy/test_trajectory_subset.cu")
ConfigureTest(LEGACY_TRAJECTORY_SUBSET_TEST "${LEGACY_TRAJECTORY_SUBSET_TEST_SRC}")

set(TRAJECTORY_DISTANCES_AND_SPEEDS_TEST_SRC
"${CMAKE_CURRENT_SOURCE_DIR}/trajectory/test_trajectory_distances_and_speeds.cu")
ConfigureTest(TRAJECTORY_DISTANCES_AND_SPEEDS_TEST "${TRAJECTORY_DISTANCES_AND_SPEEDS_TEST_SRC}")

set(DERIVE_TRAJECTORIES_TEST_SRC
"${CMAKE_CURRENT_SOURCE_DIR}/trajectory/test_derive_trajectories.cu")
ConfigureTest(DERIVE_TRAJECTORIES_TEST "${DERIVE_TRAJECTORIES_TEST_SRC}")

set(TRAJECTORY_BOUNDING_BOXES_TEST_SRC
"${CMAKE_CURRENT_SOURCE_DIR}/trajectory/test_trajectory_bounding_boxes.cu")
ConfigureTest(TRAJECTORY_BOUNDING_BOXES_TEST "${TRAJECTORY_BOUNDING_BOXES_TEST_SRC}")

set(SPATIAL_WINDOW_POINT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial_window/spatial_window_test.cpp")
ConfigureTest(SPATIAL_WINDOW_POINT_TEST "${SPATIAL_WINDOW_POINT_TEST_SRC}")
