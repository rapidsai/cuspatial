#=============================================================================
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
add_library(cuproj_benchmark_common OBJECT
    synchronization/synchronization.cpp)

target_compile_features(cuproj_benchmark_common PUBLIC cxx_std_17 cuda_std_17)

target_link_libraries(cuproj_benchmark_common
  PUBLIC benchmark::benchmark rmm::rmm cuproj)

target_compile_options(cuproj_benchmark_common
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUPROJ_CXX_FLAGS}>"
         "$<$<COMPILE_LANGUAGE:CUDA>:${CUPROJ_CUDA_FLAGS}>")

target_include_directories(cuproj_benchmark_common
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
         "$<BUILD_INTERFACE:${CUPROJ_SOURCE_DIR}>"
         "$<BUILD_INTERFACE:${CUPROJ_SOURCE_DIR}/src>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>")

function(ConfigureBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN})
  set_target_properties(${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUPROJ_BINARY_DIR}/benchmarks>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
    )
  target_link_libraries(${CMAKE_BENCH_NAME} PRIVATE benchmark::benchmark_main cuproj_benchmark_common PROJ::proj)
  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT benchmark
    DESTINATION bin/benchmarks/libcuproj
    EXCLUDE_FROM_ALL
  )
endfunction()

# This function takes in a benchmark name and benchmark source for nvbench benchmarks and handles
# setting all of the associated properties and linking to build the benchmark
function(ConfigureNVBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUPROJ_BINARY_DIR}/benchmarks>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )
  target_compile_features(${CMAKE_BENCH_NAME} PUBLIC cxx_std_17 cuda_std_17)

  target_compile_options(${CMAKE_BENCH_NAME}
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUPROJ_CXX_FLAGS}>"
           "$<$<COMPILE_LANGUAGE:CUDA>:${CUPROJ_CUDA_FLAGS}>")

  target_include_directories(${CMAKE_BENCH_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
           "$<BUILD_INTERFACE:${CUPROJ_SOURCE_DIR}>"
           "$<BUILD_INTERFACE:${CUPROJ_SOURCE_DIR}/src>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>")

  target_link_libraries(
    ${CMAKE_BENCH_NAME} PRIVATE cuproj nvbench::main
  )
  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT benchmark
    DESTINATION bin/benchmarks/libcuproj
    EXCLUDE_FROM_ALL
  )
endfunction()

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

ConfigureBench(WGS_TO_UTM_BENCH wgs_to_utm_bench.cu)
