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

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <memory>
#include <tests/utilities/column_wrapper.hpp>

static void BM_test(benchmark::State& state)
{
  for (auto _ : state) {
    cuda_event_timer raii(state, true);
  }
}

class Test : public cudf::benchmark {
};

#define DUMMY_BM_BENCHMARK_DEFINE(name)                             \
  BENCHMARK_DEFINE_F(Test, name)(::benchmark::State & state)        \
  {                                                                 \
    BM_test(state);                                                 \
  }                                                                 \
  BENCHMARK_REGISTER_F(Test, name)                                  \
    ->RangeMultiplier(32)                                           \
    ->Range(1 << 10, 1 << 30)                                       \
    ->UseManualTime()                                               \
    ->Unit(benchmark::kMillisecond);

DUMMY_BM_BENCHMARK_DEFINE(test);
