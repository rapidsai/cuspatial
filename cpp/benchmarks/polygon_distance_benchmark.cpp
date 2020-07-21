/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuspatial/polygon_distance.hpp>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <tests/utilities/column_wrapper.hpp>

#include <thrust/iterator/constant_iterator.h>

static void BM_polygon(benchmark::State& state)
{
  int32_t num_points           = state.range(1) - 1;
  int32_t num_spaces_asked     = state.range(0) - 1;
  int32_t num_spaces           = std::min(num_points, num_spaces_asked);
  int32_t num_points_per_space = num_points / num_spaces;

  auto counting_iter = thrust::counting_iterator<int32_t>();
  auto zero_iter     = thrust::make_transform_iterator(counting_iter, [](auto idx) { return 0; });

  auto space_offset_iter = thrust::make_transform_iterator(
    counting_iter, [num_points_per_space](int32_t idx) { return idx * num_points_per_space; });

  auto xs = cudf::test::fixed_width_column_wrapper<double>(zero_iter, zero_iter + num_points);
  auto ys = cudf::test::fixed_width_column_wrapper<double>(zero_iter, zero_iter + num_points);

  auto space_offsets = cudf::test::fixed_width_column_wrapper<int32_t>(
    space_offset_iter, space_offset_iter + num_spaces);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    cuspatial::directed_polygon_distance(xs, ys, space_offsets);
  }

  state.SetItemsProcessed(state.iterations() * num_points * num_points);
}

class PolygonDistanceBenchmark : public cuspatial::benchmark {
};

#define DUMMY_BM_BENCHMARK_DEFINE(name)                                          \
  BENCHMARK_DEFINE_F(PolygonDistanceBenchmark, name)(::benchmark::State & state) \
  {                                                                              \
    BM_polygon(state);                                                           \
  }                                                                              \
  BENCHMARK_REGISTER_F(PolygonDistanceBenchmark, name)                           \
    ->Ranges({{1 << 10, 1 << 14}, {1 << 10, 1 << 15}})                           \
    ->UseManualTime()                                                            \
    ->Unit(benchmark::kMillisecond);

DUMMY_BM_BENCHMARK_DEFINE(polygon);
