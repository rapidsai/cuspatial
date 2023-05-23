/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <nvbench/nvbench.cuh>

#include <cuspatial_test/geometry_generator.cuh>

#include <cuspatial/distance.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/multilinestring_range.cuh>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <memory>

using namespace cuspatial;

template <typename T>
void pairwise_linestring_distance_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  auto const num_string_pairs{static_cast<std::size_t>(state.get_int64("NumStrings"))};
  auto const num_segments_per_string{static_cast<std::size_t>(state.get_int64("NumSegmentsPerString"))};

  auto params1 = test::multilinestring_generator_parameter<T>{
    num_string_pairs, num_string_pairs, num_segments_per_string, 1.0, {0., 0.}};
  auto params2 = test::multilinestring_generator_parameter<T>{
    num_string_pairs, num_string_pairs, num_segments_per_string, 1.0, {100., 100.}};

  auto ls1 = generate_linestring_array(params1, stream);
  auto ls2 = generate_linestring_array(params2, stream);

  auto ls1range = ls1.range();
  auto ls2range = ls2.range();

  auto distances = rmm::device_uvector<T>(ls1range.size(), stream);
  auto out_it    = distances.begin();

  auto const total_points = ls1.size() + ls2.size();

  state.add_element_count(num_string_pairs, "LineStringPairs");
  state.add_element_count(total_points, "NumPoints");
  state.add_global_memory_reads<T>(total_points * 2, "CoordinatesDataSize");
  state.add_global_memory_reads<int32_t>(num_string_pairs * 2, "OffsetsDataSize");
  state.add_global_memory_writes<T>(num_string_pairs);

  state.exec(nvbench::exec_tag::sync, [&ls1range, &ls2range, &out_it](nvbench::launch& launch) {
    pairwise_linestring_distance(ls1range, ls2range, out_it);
  });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(pairwise_linestring_distance_benchmark, NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("NumStrings", {1'000, 10'000, 100'000})
  .add_int64_axis("NumSegmentsPerString", {10, 100, 1'000});
