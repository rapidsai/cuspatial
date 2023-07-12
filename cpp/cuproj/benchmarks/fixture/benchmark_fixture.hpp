/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace cuspatial {

namespace {
// memory resource factory helpers
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
}
}  // namespace

/**
 * @brief Google Benchmark fixture for libcuspatial benchmarks
 *
 * libcuspatial benchmarks should use a fixture derived from this fixture class to
 * ensure that the RAPIDS Memory Manager pool mode is used in benchmarks, which
 * eliminates memory allocation / deallocation performance overhead from the
 * benchmark.
 *
 * The SetUp and TearDown methods of this fixture initialize RMM into pool mode
 * and finalize it, respectively. These methods are called automatically by
 * Google Benchmark
 *
 * Example:
 *
 * template <class T>
 * class my_benchmark : public cuspatial::benchmark {
 * public:
 *   using TypeParam = T;
 * };
 *
 * Then:
 *
 * BENCHMARK_TEMPLATE_DEFINE_F(my_benchmark, my_test_name, int)
 *   (::benchmark::State& state) {
 *     for (auto _ : state) {
 *       // benchmark stuff
 *     }
 * }
 *
 * BENCHMARK_REGISTER_F(my_benchmark, my_test_name)->Range(128, 512);
 */
class benchmark : public ::benchmark::Fixture {
 public:
  virtual void SetUp(const ::benchmark::State& state) override
  {
    mr = make_pool();
    rmm::mr::set_current_device_resource(mr.get());  // set default resource to pool
  }

  virtual void TearDown(const ::benchmark::State& state) override
  {
    // reset default resource to the initial resource
    rmm::mr::set_current_device_resource(nullptr);
  }

  // eliminate partial override warnings (see benchmark/benchmark.h)
  void SetUp(::benchmark::State& st) override { SetUp(const_cast<const ::benchmark::State&>(st)); }
  void TearDown(::benchmark::State& st) override
  {
    TearDown(const_cast<const ::benchmark::State&>(st));
  }

  std::shared_ptr<rmm::mr::device_memory_resource> mr;
};

};  // namespace cuspatial