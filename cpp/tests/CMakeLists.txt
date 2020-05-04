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
                          cudftestutil rmm cudart cuda "${ARROW_LIB}" ${ZLIB_LIBRARIES} NVCategory
                          NVStrings nvrtc gdal)
    if(USE_NVTX)
        target_link_libraries(${CMAKE_TEST_NAME} ${NVTX_LIBRARY})
    endif(USE_NVTX)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

option(CMAKE_ENABLE_BENCHMARKS "Enable building cuSpatial benchmarks" OFF)

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    if(CMAKE_ENABLE_BENCHMARKS)
        add_executable(${CMAKE_BENCH_NAME}
                    ${CMAKE_BENCH_SRC})
        set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${CMAKE_BENCH_NAME} cuspatial cudf cudftestutil)
        set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")
        add_test(NAME ${CMAKE_BENCH_NAME} COMMAND ${CMAKE_BENCH_NAME})
    endif(CMAKE_ENABLE_BENCHMARKS)
endfunction(ConfigureBench)


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

set(LEGACY_POINT_IN_POLYGON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/pip_toy.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/pip_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial/legacy/pip_compare.cu")
ConfigureTest(LEGACY_POINT_IN_POLYGON_TEST "${LEGACY_POINT_IN_POLYGON_TEST_SRC}")

set(SHAPEFILE_POYGON_READER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/read_shapefile_polygon_test.cu")
ConfigureTest(SHAPEFILE_POLYGON_READER_TEST "${SHAPEFILE_POYGON_READER_TEST_SRC}")

set(SPATIAL_WINDOW_POINT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/query/spatial_window_test_toy.cu")
ConfigureTest(SPATIAL_WINDOW_POINT_TEST "${SPATIAL_WINDOW_POINT_TEST_SRC}")

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
