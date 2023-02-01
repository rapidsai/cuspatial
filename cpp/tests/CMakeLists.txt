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

ConfigureTest(SINUSOIDAL_PROJECTION_TEST
    spatial/sinusoidal_projection_test.cu)

ConfigureTest(HAVERSINE_TEST
    spatial/haversine_test.cpp)

ConfigureTest(HAUSDORFF_TEST
    spatial/hausdorff_test.cpp)

ConfigureTest(JOIN_POINT_IN_POLYGON_SMALL_TEST
    join/point_in_polygon_test_small.cpp)

ConfigureTest(JOIN_POINT_IN_POLYGON_LARGE_TEST
    join/point_in_polygon_test_large.cpp)

ConfigureTest(JOIN_POINT_TO_LINESTRING_SMALL_TEST
    join/point_to_nearest_linestring_test_small.cpp)

ConfigureTest(POINT_IN_POLYGON_TEST
    spatial/point_in_polygon_test.cpp)

ConfigureTest(PAIRWISE_POINT_IN_POLYGON_TEST
    spatial/pairwise_point_in_polygon_test.cpp)

ConfigureTest(POINT_QUADTREE_TEST
    indexing/point_quadtree_test.cu)

ConfigureTest(LINESTRING_BOUNDING_BOXES_TEST
    spatial/linestring_bounding_boxes_test.cu)

ConfigureTest(POLYGON_BOUNDING_BOXES_TEST
    spatial/polygon_bounding_boxes_test.cu)

ConfigureTest(POINT_DISTANCE_TEST
    spatial/point_distance_test.cpp)

ConfigureTest(POINT_LINESTRING_DISTANCE_TEST
    spatial/point_linestring_distance_test.cpp)

ConfigureTest(LINESTRING_DISTANCE_TEST
    spatial/linestring_distance_test.cpp)

ConfigureTest(POINT_LINESTRING_NEAREST_POINT_TEST
    spatial/point_linestring_nearest_points_test.cpp)

ConfigureTest(SHAPEFILE_READER_TEST
    io/shp/polygon_shapefile_reader_test.cpp)

ConfigureTest(QUADTREE_POLYGON_FILTERING_TEST
    join/quadtree_polygon_filtering_test.cu)

ConfigureTest(QUADTREE_LINESTRING_FILTERING_TEST
    join/quadtree_linestring_filtering_test.cu)

ConfigureTest(TRAJECTORY_DISTANCES_AND_SPEEDS_TEST
    trajectory/test_trajectory_distances_and_speeds.cu)

ConfigureTest(DERIVE_TRAJECTORIES_TEST
    trajectory/test_derive_trajectories.cpp)

ConfigureTest(TRAJECTORY_BOUNDING_BOXES_TEST
    trajectory/test_trajectory_bounding_boxes.cu)

ConfigureTest(SPATIAL_WINDOW_POINT_TEST
    spatial_window/spatial_window_test.cpp)

ConfigureTest(UTILITY_TEST
    utility_test/test_float_equivalent.cu
    utility_test/test_multipoint_factory.cu
)

# Experimental API
ConfigureTest(HAVERSINE_TEST_EXP
    experimental/spatial/haversine_test.cu)

ConfigureTest(POINT_DISTANCE_TEST_EXP
    experimental/spatial/point_distance_test.cu)

ConfigureTest(POINT_LINESTRING_DISTANCE_TEST_EXP
    experimental/spatial/point_linestring_distance_test.cu)

ConfigureTest(HAUSDORFF_TEST_EXP
    experimental/spatial/hausdorff_test.cu)

ConfigureTest(LINESTRING_DISTANCE_TEST_EXP
    experimental/spatial/linestring_distance_test.cu
    experimental/spatial/linestring_distance_test_medium.cu)

ConfigureTest(LINESTRING_INTERSECTION_TEST_EXP
    experimental/spatial/linestring_intersection_count_test.cu
    experimental/spatial/linestring_intersection_intermediates_remove_if_test.cu
    experimental/spatial/linestring_intersection_with_duplicates_test.cu
    experimental/spatial/linestring_intersection_test.cu)

ConfigureTest(POINT_LINESTRING_NEAREST_POINT_TEST_EXP
    experimental/spatial/point_linestring_nearest_points_test.cu)

ConfigureTest(SINUSOIDAL_PROJECTION_TEST_EXP
    experimental/spatial/sinusoidal_projection_test.cu)

ConfigureTest(POINTS_IN_RANGE_TEST_EXP
    experimental/spatial/points_in_range_test.cu)

ConfigureTest(POINT_IN_POLYGON_TEST_EXP
    experimental/spatial/point_in_polygon_test.cu)

ConfigureTest(PAIRWISE_POINT_IN_POLYGON_TEST_EXP
    experimental/spatial/pairwise_point_in_polygon_test.cu)

ConfigureTest(DERIVE_TRAJECTORIES_TEST_EXP
    experimental/trajectory/derive_trajectories_test.cu)

ConfigureTest(POINT_BOUNDING_BOXES_TEST_EXP
    experimental/spatial/point_bounding_boxes_test.cu)

ConfigureTest(POLYGON_BOUNDING_BOXES_TEST_EXP
    experimental/spatial/polygon_bounding_boxes_test.cu)

ConfigureTest(LINESTRING_BOUNDING_BOXES_TEST_EXP
    experimental/spatial/linestring_bounding_boxes_test.cu)

ConfigureTest(TRAJECTORY_DISTANCES_AND_SPEEDS_TEST_EXP
    experimental/trajectory/trajectory_distances_and_speeds_test.cu)

ConfigureTest(POINT_QUADTREE_TEST_EXP
    experimental/indexing/point_quadtree_test.cu)

ConfigureTest(OPERATOR_TEST_EXP
    experimental/operators/linestrings_test.cu)

ConfigureTest(FIND_TEST_EXP
    experimental/find/find_and_combine_segments_test.cu
    experimental/find/find_points_on_segments_test.cu
    experimental/find/find_duplicate_points_test.cu)
