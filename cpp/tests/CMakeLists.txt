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

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cuspatial cudf
                        cudftestutil rmm cudart cuda "${ARROW_LIB}" ${ZLIB_LIBRARIES}
                        nvrtc gdal)
    if(USE_NVTX)
        target_link_libraries(${CMAKE_TEST_NAME} ${NVTX_LIBRARY})
    endif(USE_NVTX)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
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

set(COORDINATE_TRANSFORM_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/coordinate_transform_test.cu")
ConfigureTest(COORDINATE_TRANSFORM_TEST "${COORDINATE_TRANSFORM_TEST_SRC}")

set(HAVERSINE_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/haversine_test.cpp")
ConfigureTest(HAVERSINE_TEST "${HAVERSINE_TEST_SRC}")

set(HAUSDORFF_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/hausdorff_test.cpp")
ConfigureTest(HAUSDORFF_TEST "${HAUSDORFF_TEST_SRC}")

set(POINT_IN_POLYGON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/point_in_polygon_test.cpp")
ConfigureTest(POINT_IN_POLYGON_TEST "${POINT_IN_POLYGON_TEST_SRC}")

set(POINT_QUADTREE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/indexing/point_quadtree_test.cu")
ConfigureTest(POINT_QUADTREE_TEST "${POINT_QUADTREE_TEST_SRC}")

set(POLYLINE_BBOX_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/polyline_bbox_test.cu")
ConfigureTest(POLYLINE_BBOX_TEST "${POLYLINE_BBOX_TEST_SRC}")

set(POLYGON_BBOX_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/polygon_bbox_test.cu")
ConfigureTest(POLYGON_BBOX_TEST "${POLYGON_BBOX_TEST_SRC}")

set(SHAPEFILE_READER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/shp/polygon_shapefile_reader_test.cpp")
ConfigureTest(SHAPEFILE_READER_TEST "${SHAPEFILE_READER_TEST_SRC}")

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
