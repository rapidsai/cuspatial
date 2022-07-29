/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cuspatial/distance/hausdorff.hpp>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf_test/column_wrapper.hpp>

#include <thrust/iterator/constant_iterator.h>

static void BM_hausdorff(benchmark::State& state)
{
  int32_t num_spaces           = state.range(0) - 1;
  int32_t num_points_per_space = state.range(1) - 1;
  int32_t num_points           = num_points_per_space * num_spaces;

  auto zero_iter = thrust::make_constant_iterator(0);

  auto space_offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [num_points_per_space](int32_t idx) { return idx * num_points_per_space; });

  auto xs = cudf::test::fixed_width_column_wrapper<double>(zero_iter, zero_iter + num_points);
  auto ys = cudf::test::fixed_width_column_wrapper<double>(zero_iter, zero_iter + num_points);

  auto space_offsets = cudf::test::fixed_width_column_wrapper<int32_t>(
    space_offset_iter, space_offset_iter + num_spaces);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    cuspatial::directed_hausdorff_distance(xs, ys, space_offsets);
  }

  state.SetItemsProcessed(state.iterations() * num_points * num_points);
}

class HausdorffBenchmark : public cuspatial::benchmark {
  virtual void SetUp(const ::benchmark::State& state) override
  {
    mr = std::make_shared<rmm::mr::cuda_memory_resource>();
    rmm::mr::set_current_device_resource(mr.get());  // set default resource to cuda
  }
};

#define DUMMY_BM_BENCHMARK_DEFINE(name)                                    \
  BENCHMARK_DEFINE_F(HausdorffBenchmark, name)(::benchmark::State & state) \
  {                                                                        \
    BM_hausdorff(state);                                                   \
  }                                                                        \
  BENCHMARK_REGISTER_F(HausdorffBenchmark, name)                           \
    ->Ranges({{1 << 5, 1 << 13}, {1 << 2, 1 << 7}})                        \
    ->UseManualTime()                                                      \
    ->Unit(benchmark::kMillisecond);

DUMMY_BM_BENCHMARK_DEFINE(hausdorff);
