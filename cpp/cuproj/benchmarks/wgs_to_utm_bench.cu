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

#include <cuproj_test/coordinate_generator.cuh>

#include <cuproj/projection_factories.hpp>

#include <cuspatial/geometry/vec_2d.hpp>

#include <benchmarks/fixture/rmm_pool_raii.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/host_vector.h>

#include <type_traits>

template <typename T>
using coordinate = typename cuspatial::vec_2d<T>;

template <typename T>
void wgs_to_utm_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;

  auto const num_points{static_cast<std::size_t>(state.get_int64("NumPoints"))};
  auto const grid_size{static_cast<std::size_t>(std::sqrt(num_points))};

  // Sydney Harbour
  coordinate<T> min_corner{-33.9, 151.2};
  coordinate<T> max_corner{-33.7, 151.3};
  char const* epsg_src = "EPSG:4326";
  char const* epsg_dst = "EPSG:32756";

  auto input = cuproj_test::make_grid_array<coordinate<T>, rmm::device_vector<coordinate<T>>>(
    min_corner, max_corner, grid_size, grid_size);

  rmm::device_vector<coordinate<T>> output(input.size());

  auto proj = cuproj::make_projection<coordinate<T>>(epsg_src, epsg_dst);

  state.add_element_count(grid_size * grid_size, "NumPoints");

  state.exec(nvbench::exec_tag::sync, [&proj, &input, &output](nvbench::launch& launch) {
    proj.transform(input.begin(),
                   input.end(),
                   output.begin(),
                   cuproj::direction::FORWARD,
                   rmm::cuda_stream_default);
  });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(wgs_to_utm_benchmark, NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("NumPoints", {100, 10'000, 100'000, 100'000'000});
