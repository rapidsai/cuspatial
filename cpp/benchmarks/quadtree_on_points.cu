/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial/experimental/point_quadtree.cuh>
#include <cuspatial/vec_2d.hpp>

#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <nvbench/nvbench.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <rmm/exec_policy.hpp>
#include <thrust/host_vector.h>

using namespace cuspatial;

template <typename T>
auto generate_rects(T const size)
{
  auto const phi = static_cast<T>((1 + std::sqrt(5)) * .5);

  vec_2d<T> tl{T{0}, T{0}};
  vec_2d<T> br{size, size};
  vec_2d<T> area = br - tl;

  std::size_t num_points = 0;
  std::vector<std::tuple<std::size_t, vec_2d<T>, vec_2d<T>>> rects{};

  do {
    switch (rects.size() % 4) {
      case 0: br.x = tl.x - (tl.x - br.x) / phi; break;
      case 1: br.y = tl.y - (tl.y - br.y) / phi; break;
      case 2: tl.x = tl.x + (br.x - tl.x) / phi; break;
      case 3: tl.y = tl.y + (br.y - tl.y) / phi; break;
    }

    area = br - tl;

    auto num_points_in_rect = static_cast<std::size_t>(std::sqrt(area.x * area.y * 1'000'000));

    rects.push_back(std::make_tuple(num_points_in_rect, tl, br));

    num_points += num_points_in_rect;
  } while (area.x > 1 && area.y > 1);

  return std::make_pair(num_points, std::move(rects));
}

/**
 * @brief Generate a random point within a window of [minXY, maxXY]
 */
template <typename T>
vec_2d<T> random_point(vec_2d<T> minXY, vec_2d<T> maxXY)
{
  auto x = minXY.x + (maxXY.x - minXY.x) * rand() / static_cast<T>(RAND_MAX);
  auto y = minXY.y + (maxXY.y - minXY.y) * rand() / static_cast<T>(RAND_MAX);
  return vec_2d<T>{x, y};
}

template <typename T>
std::pair<std::size_t, std::vector<vec_2d<T>>> generate_points(T const size)
{
  auto const [total_points, rects] = generate_rects(size);

  std::size_t point_offset{0};
  std::vector<vec_2d<T>> h_points(total_points);
  for (auto const& rect : rects) {
    auto const num_points_in_rect = std::get<0>(rect);
    auto const tl                 = std::get<1>(rect);
    auto const br                 = std::get<2>(rect);
    auto points_begin             = h_points.begin() + point_offset;
    auto points_end               = points_begin + num_points_in_rect;
    std::generate(points_begin, points_end, [&]() { return random_point<T>(tl, br); });
    point_offset += num_points_in_rect;
  }
  return {total_points, h_points};
}

template <typename T>
void quadtree_on_points_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  auto const [total_points, h_points] =
    generate_points(static_cast<T>(state.get_float64("Bounding box size")));

  auto const max_size = static_cast<int32_t>(total_points / std::pow(4, 4));

  rmm::device_vector<vec_2d<T>> d_points(h_points);
  auto const vertex_1_itr = thrust::min_element(
    thrust::device, d_points.begin(), d_points.end(), [] __device__(auto const& a, auto const& b) {
      return a.x <= b.x && a.y <= b.y;
    });
  auto const vertex_1 = h_points[thrust::distance(d_points.begin(), vertex_1_itr)];

  auto const vertex_2_itr = thrust::max_element(
    thrust::device, d_points.begin(), d_points.end(), [] __device__(auto const& a, auto const& b) {
      return a.x >= b.x && a.y >= b.y;
    });
  auto const vertex_2 = h_points[thrust::distance(d_points.begin(), vertex_2_itr)];

  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;

  state.add_element_count(max_size, "Split threshold");
  state.add_element_count(total_points, "Total Points");
  state.exec(nvbench::exec_tag::sync,
             [&d_points, &vertex_1, &vertex_2, max_size](nvbench::launch& launch) {
               quadtree_on_points(
                 d_points.begin(), d_points.end(), vertex_1, vertex_2, T{-1}, int8_t{15}, max_size);
             });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(quadtree_on_points_benchmark, NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"FP type"})
  .add_float64_axis("Bounding box size", {1'000, 10'000, 100'000});
