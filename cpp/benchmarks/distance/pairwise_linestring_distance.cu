/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <benchmarks/fixture/rmm_pool_raii.hpp>

#include <cuspatial_test/geometry_generator.cuh>

#include <cuspatial/distance.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>

using namespace cuspatial;

template <typename T>
void pairwise_linestring_distance_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  auto const num_pairs{static_cast<std::size_t>(state.get_int64("NumPairs"))};
  auto const num_linestrings_per_multilinestring{
    static_cast<std::size_t>(state.get_int64("NumLineStringsPerMultiLineString"))};
  auto const num_segments_per_linestring{
    static_cast<std::size_t>(state.get_int64("NumSegmentsPerLineString"))};

  auto params1 = test::multilinestring_generator_parameter<T>{
    num_pairs, num_linestrings_per_multilinestring, num_segments_per_linestring, 1.0, {0., 0.}};
  auto params2 = test::multilinestring_generator_parameter<T>{num_pairs,
                                                              num_linestrings_per_multilinestring,
                                                              num_segments_per_linestring,
                                                              1.0,
                                                              {100000., 100000.}};

  auto ls1 = generate_multilinestring_array(params1, stream);
  auto ls2 = generate_multilinestring_array(params2, stream);

  auto ls1range = ls1.range();
  auto ls2range = ls2.range();

  auto output = rmm::device_uvector<T>(num_pairs, stream);
  auto out_it = output.begin();

  auto const total_points = params1.num_points() + params2.num_points();

  state.add_element_count(num_pairs, "NumPairs");
  state.add_element_count(total_points, "NumPoints");

  state.add_global_memory_reads<T>(total_points * 2, "CoordinatesDataSize");
  state.add_global_memory_reads<int32_t>(params1.num_multilinestrings +
                                           params2.num_multilinestrings +
                                           params1.num_linestrings() + params2.num_linestrings(),
                                         "OffsetsDataSize");
  state.add_global_memory_writes<T>(num_pairs);

  state.exec(nvbench::exec_tag::sync, [&ls1range, &ls2range, &out_it](nvbench::launch& launch) {
    pairwise_linestring_distance(ls1range, ls2range, out_it);
  });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(pairwise_linestring_distance_benchmark, NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("NumPairs", {1'000, 10'000, 100'000})
  .add_int64_axis("NumLineStringsPerMultiLineString", {1'000, 10'000, 100'000})
  .add_int64_axis("NumSegmentsPerLineString", {10, 100, 1'000});
