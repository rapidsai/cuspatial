#=============================================================================
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    target_compile_options(${CMAKE_TEST_NAME}
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_FLAGS}>")
    target_include_directories(${CMAKE_TEST_NAME}
                PRIVATE "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}>"
                        "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}/src>")
    set_target_properties(
        ${CMAKE_TEST_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUSPATIAL_BINARY_DIR}/gtests>"
                   INSTALL_RPATH "\$ORIGIN/../../../lib"
    )
    target_link_libraries(${CMAKE_TEST_NAME} GTest::gtest_main GTest::gmock_main cudf::cudftestutil cuspatial)
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    install(
        TARGETS ${CMAKE_TEST_NAME}
        COMPONENT testing
        DESTINATION bin/gtests/libcuspatial
        EXCLUDE_FROM_ALL
    )
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

ConfigureTest(CUBIC_SPLINE_TEST
    interpolate/cubic_spline_test.cpp)

ConfigureTest(COORDINATE_TRANSFORM_TEST
    spatial/coordinate_transform_test.cu)

ConfigureTest(HAVERSINE_TEST
    spatial/haversine_test.cpp)

ConfigureTest(HAUSDORFF_TEST
    spatial/hausdorff_test.cpp)

ConfigureTest(JOIN_POINT_IN_POLYGON_SMALL_TEST
    join/point_in_polygon_test_small.cpp)

ConfigureTest(JOIN_POINT_IN_POLYGON_LARGE_TEST
    join/point_in_polygon_test_large.cpp)

ConfigureTest(JOIN_POINT_TO_POLYLINE_SMALL_TEST
    join/point_to_nearest_polyline_test_small.cpp)

ConfigureTest(POINT_IN_POLYGON_TEST
    spatial/point_in_polygon_test.cpp)

ConfigureTest(POINT_QUADTREE_TEST
    indexing/point_quadtree_test.cu)

ConfigureTest(POLYLINE_BBOX_TEST
    spatial/polyline_bbox_test.cu)

ConfigureTest(POLYGON_BBOX_TEST
    spatial/polygon_bbox_test.cu)

ConfigureTest(LINESTRING_DISTANCE_TEST
    spatial/linestring_distance_test.cpp spatial/linestring_distance_test_medium.cpp)

ConfigureTest(SHAPEFILE_READER_TEST
    io/shp/polygon_shapefile_reader_test.cpp)

ConfigureTest(QUADTREE_POLYGON_FILTERING_TEST
    join/quadtree_polygon_filtering_test.cu)

ConfigureTest(QUADTREE_POLYLINE_FILTERING_TEST
    join/quadtree_polyline_filtering_test.cu)

ConfigureTest(TRAJECTORY_DISTANCES_AND_SPEEDS_TEST
    trajectory/test_trajectory_distances_and_speeds.cu)

ConfigureTest(DERIVE_TRAJECTORIES_TEST
    trajectory/test_derive_trajectories.cu)

ConfigureTest(TRAJECTORY_BOUNDING_BOXES_TEST
    trajectory/test_trajectory_bounding_boxes.cu)

ConfigureTest(SPATIAL_WINDOW_POINT_TEST
    spatial_window/spatial_window_test.cpp)

# Experimental API
ConfigureTest(HAVERSINE_TEST_EXP
    experimental/spatial/haversine_test.cu)

ConfigureTest(LINESTRING_DISTANCE_TEST_EXP
    experimental/spatial/linestring_distance_test.cu)

ConfigureTest(COORDINATE_TRANSFORM_TEST_EXP
    experimental/spatial/coordinate_transform_test.cu)
