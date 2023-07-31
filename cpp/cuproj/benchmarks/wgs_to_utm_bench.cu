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
#include <cuspatial/geometry/vec_2d.hpp>

#include <cuproj_test/convert_coordinates.hpp>
#include <cuproj_test/coordinate_generator.cuh>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>

#include <type_traits>

template <typename T>
using coordinate = typename cuspatial::vec_2d<T>;

static char const* epsg_src = "EPSG:4326";
static char const* epsg_dst = "EPSG:32756";

template <typename T>
auto make_input(std::size_t grid_side)
{
  // Sydney Harbour
  coordinate<T> min_corner{-33.9, 151.2};
  coordinate<T> max_corner{-33.7, 151.3};

  auto input = cuproj_test::make_grid_array<coordinate<T>, rmm::device_vector<coordinate<T>>>(
    min_corner, max_corner, grid_side, grid_side);

  return input;
}

template <typename T>
static void cuproj_wgs_to_utm_benchmark(benchmark::State& state)
{
  auto const grid_side{static_cast<std::size_t>(state.range(0))};

  auto input = make_input<T>(grid_side);

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

void proj_wgs_to_utm_benchmark(benchmark::State& state)
{
  using T = double;
  auto const grid_side{static_cast<std::size_t>(state.range(0))};

  auto d_input = make_input<T>(grid_side);
  auto input   = thrust::host_vector<coordinate<T>>(d_input);

  std::vector<PJ_COORD> pj_input(input.size());
  cuproj_test::convert_coordinates(input, pj_input);

  PJ_CONTEXT* C = proj_context_create();
  PJ* P         = proj_create_crs_to_crs(C, epsg_src, epsg_dst, nullptr);

  for (auto _ : state) {
    state.PauseTiming();
    cuproj_test::convert_coordinates(input, pj_input);
    state.ResumeTiming();
    proj_trans_array(P, PJ_FWD, pj_input.size(), pj_input.data());
  }

  state.SetItemsProcessed(grid_side * grid_side * state.iterations());
}

class proj_utm_benchmark : public ::benchmark::Fixture {};

BENCHMARK_DEFINE_F(proj_utm_benchmark, forward_double)(::benchmark::State& state)
{
  proj_wgs_to_utm_benchmark(state);
}
BENCHMARK_REGISTER_F(proj_utm_benchmark, forward_double)
  ->Range(8, 16384)
  ->Unit(benchmark::kMillisecond);

class cuproj_utm_benchmark : public cuspatial::benchmark {};

#define UTM_CUPROJ_BENCHMARK_DEFINE(name, type)                              \
  BENCHMARK_DEFINE_F(cuproj_utm_benchmark, name)(::benchmark::State & state) \
  {                                                                          \
    cuproj_wgs_to_utm_benchmark<type>(state);                                \
  }                                                                          \
  BENCHMARK_REGISTER_F(cuproj_utm_benchmark, name)                           \
    ->Range(8, 16384)                                                        \
    ->UseManualTime()                                                        \
    ->Unit(benchmark::kMillisecond);

UTM_CUPROJ_BENCHMARK_DEFINE(forward_float, float);
UTM_CUPROJ_BENCHMARK_DEFINE(forward_double, double);
