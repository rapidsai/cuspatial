/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <benchmark/benchmark.h>

#include <tests/utilities/column_wrapper.hpp>

#include <cuspatial/hausdorff.hpp>

#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <memory>
#include <random>

template <typename T>
T random_number(T min, T max)
{
  static unsigned seed = 73311337;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

static void BM_hausdorff(benchmark::State& state)
{
  int32_t num_spaces = 10;
  int32_t num_points_per_space = state.range(0);
  int32_t num_points = num_points_per_space * num_spaces;

  auto counting_iter = thrust::counting_iterator<int32_t>();
  auto random_double_iter = thrust::make_transform_iterator(
    counting_iter,
    [](auto idx){ return random_number<double>(-100, 100); });

  auto num_points_per_space_iter = thrust::make_transform_iterator(
    counting_iter,
    [num_points_per_space](int32_t idx){ return num_points_per_space; });

  auto xs = cudf::test::fixed_width_column_wrapper<double>(
    random_double_iter,
    random_double_iter + num_points);

  auto ys = cudf::test::fixed_width_column_wrapper<double>(
    random_double_iter,
    random_double_iter + num_points);

  auto points_per_space = cudf::test::fixed_width_column_wrapper<int32_t>(
    num_points_per_space_iter,
    num_points_per_space_iter + num_spaces);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto x = cuspatial::directed_hausdorff_distance(xs, ys, points_per_space);
  }
}

class HausdorffBenchmark : public cudf::benchmark {
};

#define DUMMY_BM_BENCHMARK_DEFINE(name)                                    \
  BENCHMARK_DEFINE_F(HausdorffBenchmark, name)(::benchmark::State & state) \
  {                                                                        \
    BM_hausdorff(state);                                                   \
  }                                                                        \
  BENCHMARK_REGISTER_F(HausdorffBenchmark, name)                           \
    ->RangeMultiplier(32)                                                  \
    ->Range(1 << 10, 1 << 30)                                              \
    ->UseManualTime()                                                      \
    ->Unit(benchmark::kMillisecond);

DUMMY_BM_BENCHMARK_DEFINE(hausdorff);
