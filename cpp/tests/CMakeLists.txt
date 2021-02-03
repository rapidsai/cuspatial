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

# Find or install CTest
include(CTest)

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

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
    target_compile_options(${CMAKE_TEST_NAME}
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_FLAGS}>")
    target_include_directories(${CMAKE_TEST_NAME}
                PRIVATE "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}>"
                        "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}/src>")
    set_target_properties(${CMAKE_TEST_NAME}
        PROPERTIES CXX_STANDARD              14
                   CXX_STANDARD_REQUIRED     ON
                   CUDA_STANDARD             14
                   CUDA_STANDARD_REQUIRED    ON
                   POSITION_INDEPENDENT_CODE ON
                   RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUSPATIAL_BINARY_DIR}/gtests>")
    target_link_libraries(${CMAKE_TEST_NAME} cudftestutil cuspatial)
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

set(CARTESIAN_PRODUCT_GROUP_INDEX_ITERATOR_TEST_SRC
   "${CMAKE_CURRENT_SOURCE_DIR}/spatial/cartesian_product_group_index_iterator_test.cpp")
ConfigureTest(CARTESIAN_PRODUCT_GROUP_INDEX_ITERATOR_TEST "${CARTESIAN_PRODUCT_GROUP_INDEX_ITERATOR_TEST_SRC}")

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

set(JOIN_POINT_IN_POLYGON_SMALL_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/point_in_polygon_test_small.cpp")
ConfigureTest(JOIN_POINT_IN_POLYGON_SMALL_TEST "${JOIN_POINT_IN_POLYGON_SMALL_TEST_SRC}")

set(JOIN_POINT_IN_POLYGON_LARGE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/point_in_polygon_test_large.cpp")
ConfigureTest(JOIN_POINT_IN_POLYGON_LARGE_TEST "${JOIN_POINT_IN_POLYGON_LARGE_TEST_SRC}")

set(JOIN_POINT_TO_POLYLINE_SMALL_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/point_to_nearest_polyline_test_small.cpp")
ConfigureTest(JOIN_POINT_TO_POLYLINE_SMALL_TEST "${JOIN_POINT_TO_POLYLINE_SMALL_TEST_SRC}")

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

set(QUADTREE_POLYGON_FILTERING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/quadtree_polygon_filtering_test.cu")
ConfigureTest(QUADTREE_POLYGON_FILTERING_TEST "${QUADTREE_POLYGON_FILTERING_SRC}")

set(QUADTREE_POLYLINE_FILTERING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/quadtree_polyline_filtering_test.cu")
ConfigureTest(QUADTREE_POLYLINE_FILTERING_TEST "${QUADTREE_POLYLINE_FILTERING_SRC}")

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
