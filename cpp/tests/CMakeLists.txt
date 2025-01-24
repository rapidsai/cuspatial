#=============================================================================
# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

# cudftestutil_impl is an interface source library, this empty object 
# library is used to speed-up compilation and linking against it, 
# otherwise we pay the non-trivial compilation cost repeatedly for each
# test executable
add_library(cuspatial_test_common OBJECT test_common.cpp)

target_compile_features(cuspatial_test_common PUBLIC cxx_std_17 cuda_std_17)

set_target_properties(cuspatial_test_common
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUSPATIAL_BINARY_DIR}/tests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(cuspatial_test_common
    PUBLIC cudf::cudftestutil GTest::gtest GTest::gmock 
    PRIVATE cudf::cudftestutil_impl)

target_compile_options(cuspatial_test_common PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_FLAGS}>"
"$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_FLAGS}>")

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
                   CXX_STANDARD 17
                   CXX_STANDARD_REQUIRED ON
                   CUDA_STANDARD 17
                   CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(${CMAKE_TEST_NAME} GTest::gtest_main GTest::gmock_main ranger::ranger cudf::cudftestutil cuspatial cuspatial_test_common)
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

# index
ConfigureTest(POINT_QUADTREE_TEST
    index/point_quadtree_test.cpp)

# join
ConfigureTest(JOIN_QUADTREE_AND_BOUNDING_BOXES_TEST
    join/join_quadtree_and_bounding_boxes_test.cpp)

ConfigureTest(JOIN_POINT_TO_NEAREST_LINESTRING_TEST
    join/quadtree_point_to_nearest_linestring_test.cpp)

ConfigureTest(JOIN_POINT_IN_POLYGON_TEST
    join/quadtree_point_in_polygon_test.cpp)

# projection
ConfigureTest(SINUSOIDAL_PROJECTION_TEST
    projection/sinusoidal_projection_test.cpp)

# bounding boxes
ConfigureTest(LINESTRING_BOUNDING_BOXES_TEST
    bounding_boxes/linestring_bounding_boxes_test.cpp)

ConfigureTest(POLYGON_BOUNDING_BOXES_TEST
    bounding_boxes/polygon_bounding_boxes_test.cpp)

# distance
ConfigureTest(HAVERSINE_TEST
    distance/haversine_test.cpp)

ConfigureTest(HAUSDORFF_TEST
    distance/hausdorff_test.cpp)

ConfigureTest(POINT_DISTANCE_TEST
    distance/point_distance_test.cpp)

ConfigureTest(POINT_LINESTRING_DISTANCE_TEST
    distance/point_linestring_distance_test.cpp)

ConfigureTest(LINESTRING_DISTANCE_TEST
    distance/linestring_distance_test.cpp)

ConfigureTest(POINT_POLYGON_DISTANCE_TEST
    distance/point_polygon_distance_test.cpp)

ConfigureTest(LINESTRING_POLYGON_DISTANCE_TEST
    distance/linestring_polygon_distance_test.cpp)

ConfigureTest(POLYGON_DISTANCE_TEST
    distance/polygon_distance_test.cpp)

# equality
ConfigureTest(PAIRWISE_MULTIPOINT_EQUALS_COUNT_TEST
    equality/pairwise_multipoint_equals_count_test.cpp)

# intersection
ConfigureTest(LINESTRING_INTERSECTION_TEST
    intersection/linestring_intersection_test.cpp)

# nearest points
ConfigureTest(POINT_LINESTRING_NEAREST_POINT_TEST
    nearest_points/point_linestring_nearest_points_test.cpp)

# point in polygon
ConfigureTest(POINT_IN_POLYGON_TEST
    point_in_polygon/point_in_polygon_test.cpp)

ConfigureTest(PAIRWISE_POINT_IN_POLYGON_TEST
    point_in_polygon/pairwise_point_in_polygon_test.cpp)

# points in range
ConfigureTest(POINTS_IN_RANGE_TEST
    points_in_range/points_in_range_test.cpp)

# trajectory
ConfigureTest(TRAJECTORY_DISTANCES_AND_SPEEDS_TEST
    trajectory/test_trajectory_distances_and_speeds.cu)

ConfigureTest(DERIVE_TRAJECTORIES_TEST
    trajectory/test_derive_trajectories.cpp)

ConfigureTest(TRAJECTORY_BOUNDING_BOXES_TEST
    trajectory/test_trajectory_bounding_boxes.cu)

# utility
ConfigureTest(UTILITY_TEST
    utility_test/test_float_equivalent.cu
    utility_test/test_multipoint_factory.cu
    utility_test/test_geometry_generators.cu
)

# find / intersection util
ConfigureTest(FIND_TEST_EXP
    find/find_and_combine_segments_test.cu
    find/find_points_on_segments_test.cu
    find/find_duplicate_points_test.cu)

# index
ConfigureTest(POINT_QUADTREE_TEST_EXP
    index/point_quadtree_test.cu)

# join
ConfigureTest(JOIN_QUADTREE_AND_BOUNDING_BOXES_TEST_EXP
    join/join_quadtree_and_bounding_boxes_test.cu)

ConfigureTest(JOIN_POINT_IN_POLYGON_SMALL_TEST_EXP
    join/quadtree_point_in_polygon_test_small.cu)

ConfigureTest(JOIN_POINT_IN_POLYGON_LARGE_TEST_EXP
    join/quadtree_point_in_polygon_test_large.cu)

ConfigureTest(JOIN_POINT_IN_POLYGON_OOM_TEST_EXP
    join/quadtree_point_in_polygon_test_oom.cu)

ConfigureTest(JOIN_POINT_TO_LINESTRING_SMALL_TEST_EXP
    join/quadtree_point_to_nearest_linestring_test_small.cu)

# operators
ConfigureTest(OPERATOR_TEST_EXP
    operators/linestrings_test.cu)

# projection
ConfigureTest(SINUSOIDAL_PROJECTION_TEST_EXP
    projection/sinusoidal_projection_test.cu)

# range
ConfigureTest(RANGE_TEST_EXP
    range/multipoint_range_test.cu
    range/multilinestring_range_test.cu
    range/multipolygon_range_test.cu)

# bounding boxes
ConfigureTest(POINT_BOUNDING_BOXES_TEST_EXP
    bounding_boxes/point_bounding_boxes_test.cu)

ConfigureTest(POLYGON_BOUNDING_BOXES_TEST_EXP
    bounding_boxes/polygon_bounding_boxes_test.cu)

ConfigureTest(LINESTRING_BOUNDING_BOXES_TEST_EXP
    bounding_boxes/linestring_bounding_boxes_test.cu)

# distance
ConfigureTest(HAUSDORFF_TEST_EXP
    distance/hausdorff_test.cu)

ConfigureTest(HAVERSINE_TEST_EXP
    distance/haversine_test.cu)

ConfigureTest(POINT_DISTANCE_TEST_EXP
    distance/point_distance_test.cu)

ConfigureTest(POINT_LINESTRING_DISTANCE_TEST_EXP
    distance/point_linestring_distance_test.cu)

ConfigureTest(POINT_POLYGON_DISTANCE_TEST_EXP
    distance/point_polygon_distance_test.cu)

ConfigureTest(LINESTRING_POLYGON_DISTANCE_TEST_EXP
    distance/linestring_polygon_distance_test.cu)

ConfigureTest(LINESTRING_DISTANCE_TEST_EXP
    distance/linestring_distance_test.cu
    distance/linestring_distance_test_medium.cu)

ConfigureTest(POLYGON_DISTANCE_TEST_EXP
    distance/polygon_distance_test.cu)

# equality
ConfigureTest(PAIRWISE_MULTIPOINT_EQUALS_COUNT_TEST_EXP
    equality/pairwise_multipoint_equals_count_test.cu)

# intersection
ConfigureTest(LINESTRING_INTERSECTION_TEST_EXP
    intersection/linestring_intersection_count_test.cu
    intersection/linestring_intersection_intermediates_remove_if_test.cu
    intersection/linestring_intersection_with_duplicates_test.cu
    intersection/linestring_intersection_test.cu
    intersection/linestring_intersection_large_test.cu)

# nearest points
ConfigureTest(POINT_LINESTRING_NEAREST_POINT_TEST_EXP
    nearest_points/point_linestring_nearest_points_test.cu)

# point in polygon
ConfigureTest(POINT_IN_POLYGON_TEST_EXP
    point_in_polygon/point_in_polygon_test.cu)

ConfigureTest(PAIRWISE_POINT_IN_POLYGON_TEST_EXP
    point_in_polygon/pairwise_point_in_polygon_test.cu)

# points in range
ConfigureTest(POINTS_IN_RANGE_TEST_EXP
    points_in_range/points_in_range_test.cu)

# trajectory
ConfigureTest(DERIVE_TRAJECTORIES_TEST_EXP
    trajectory/derive_trajectories_test.cu)

ConfigureTest(TRAJECTORY_DISTANCES_AND_SPEEDS_TEST_EXP
    trajectory/trajectory_distances_and_speeds_test.cu)
