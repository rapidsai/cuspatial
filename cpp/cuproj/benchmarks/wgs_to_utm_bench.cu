/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuproj/projection_factories.hpp>

#include <cuproj_test/coordinate_generator.cuh>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cuspatial/geometry/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>

#include <type_traits>

template <typename T>
using coordinate = typename cuspatial::vec_2d<T>;

template <typename T>
static void wgs_to_utm_benchmark(benchmark::State& state)
{
  auto const grid_side{static_cast<std::size_t>(state.range(0))};

  // Sydney Harbour
  coordinate<T> min_corner{-33.9, 151.2};
  coordinate<T> max_corner{-33.7, 151.3};
  char const* epsg_src = "EPSG:4326";
  char const* epsg_dst = "EPSG:32756";

  auto input = cuproj_test::make_grid_array<coordinate<T>, rmm::device_vector<coordinate<T>>>(
    min_corner, max_corner, grid_side, grid_side);

  rmm::device_vector<coordinate<T>> output(input.size());

  auto proj = cuproj::make_projection<coordinate<T>>(epsg_src, epsg_dst);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    proj.transform(input.begin(),
                   input.end(),
                   output.begin(),
                   cuproj::direction::FORWARD,
                   rmm::cuda_stream_default);
  }

  state.SetItemsProcessed(grid_side * grid_side * state.iterations());
}

class UtmBenchmark : public cuspatial::benchmark {
  void SetUp(const ::benchmark::State& state) override
  {
    mr = std::make_shared<rmm::mr::cuda_memory_resource>();
    rmm::mr::set_current_device_resource(mr.get());  // set default resource to cuda
  }
  void SetUp(::benchmark::State& st) override { SetUp(const_cast<const ::benchmark::State&>(st)); }
};

#define UTM_BENCHMARK_DEFINE(type)                                   \
  BENCHMARK_DEFINE_F(UtmBenchmark, type)(::benchmark::State & state) \
  {                                                                  \
    wgs_to_utm_benchmark<type>(state);                               \
  }                                                                  \
  BENCHMARK_REGISTER_F(UtmBenchmark, type)                           \
    ->Range(8, 16384)                                                \
    ->UseManualTime()                                                \
    ->Unit(benchmark::kMillisecond);

UTM_BENCHMARK_DEFINE(float);
UTM_BENCHMARK_DEFINE(double);
