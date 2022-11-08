#=============================================================================
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

# Use an OBJECT library so we only compile common source files only once
add_library(cuspatial_benchmark_common OBJECT
    synchronization/synchronization.cpp)

target_compile_features(cuspatial_benchmark_common PUBLIC cxx_std_17 cuda_std_17)

target_link_libraries(cuspatial_benchmark_common
    PUBLIC benchmark::benchmark
           cudf::cudftestutil
           cuspatial)

target_compile_options(cuspatial_benchmark_common
             PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUSPATIAL_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUSPATIAL_CUDA_FLAGS}>")

target_include_directories(cuspatial_benchmark_common
             PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
                    "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}>"
                    "$<BUILD_INTERFACE:${CUSPATIAL_SOURCE_DIR}/src>")

function(ConfigureBench CMAKE_BENCH_NAME)
    add_executable(${CMAKE_BENCH_NAME} ${ARGN})
    set_target_properties(${CMAKE_BENCH_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUSPATIAL_BINARY_DIR}/benchmarks>"
                   INSTALL_RPATH "\$ORIGIN/../../../lib"
        )
    target_link_libraries(${CMAKE_BENCH_NAME} PRIVATE benchmark::benchmark_main cuspatial_benchmark_common)
    install(
        TARGETS ${CMAKE_BENCH_NAME}
        COMPONENT benchmark
        DESTINATION bin/benchmarks/libcuspatial
        EXCLUDE_FROM_ALL
    )
endfunction()

# This function takes in a benchmark name and benchmark source for nvbench benchmarks and handles
# setting all of the associated properties and linking to build the benchmark
function(ConfigureNVBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUSPATIAL_BINARY_DIR}/benchmarks>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )
  target_link_libraries(
    ${CMAKE_BENCH_NAME} PRIVATE cuspatial_benchmark_common nvbench::main
  )
  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT benchmark
    DESTINATION bin/benchmarks/libcuspatial
    EXCLUDE_FROM_ALL
  )
endfunction()

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - hausdorff benchmark ---------------------------------------------------------------------------

ConfigureBench(HAUSDORFF_BENCH
    hausdorff_benchmark.cpp)

ConfigureNVBench(DISTANCES_BENCH
    pairwise_linestring_distance.cu)

ConfigureNVBench(POINTS_IN_RANGE_BENCH
    points_in_range.cu)

ConfigureNVBench(POINT_IN_POLYGON_BENCH
    point_in_polygon.cu)

ConfigureNVBench(QUADTREE_ON_POINTS_BENCH
    quadtree_on_points.cu)

ConfigureNVBench(FLOATING_POINT_EQUALITY_BENCH
    floating_point_equality.cu)
