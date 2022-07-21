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

#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <benchmarks/utility/random.cuh>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/spatial_window.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

#include <memory>

using namespace cuspatial;

/**
 * @brief Helper to generate random points within a rectangular window
 *
 * @p begin and @p end must be iterators to device-accessible memory
 *
 * @tparam PointsIter The type of the iterator to the output points container
 * @tparam T The floating point type for the coordinates
 * @param begin The start of the range of points to generate
 * @param end The end of the range of points to generate
 *
 * @param window_min the lower left window corner
 * @param window_max the upper right window corner
 *
 */
template <class PointsIter, typename T>
void generate_points(PointsIter begin, PointsIter end, vec_2d<T> window_min, vec_2d<T> window_max)
{
  auto engine_x = deterministic_engine(std::distance(begin, end));
  auto engine_y = deterministic_engine(2 * std::distance(begin, end));

  auto x_dist = make_uniform_dist(window_min.x, window_max.x);
  auto y_dist = make_uniform_dist(window_min.y, window_max.y);

  auto x_gen = value_generator{window_min.x, window_max.x, engine_x, x_dist};
  auto y_gen = value_generator{window_min.y, window_max.y, engine_y, y_dist};

  thrust::tabulate(rmm::exec_policy(), begin, end, [x_gen, y_gen] __device__(size_t n) mutable {
    return vec_2d<T>{x_gen(n), y_gen(n)};
  });
}

template <typename T>
void points_in_spatial_window_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;

  auto const num_points{state.get_int64("NumPoints")};

  auto window_min = vec_2d<T>{-100, -100};
  auto window_max = vec_2d<T>{100, 100};

  auto range_min = vec_2d<T>{-200, -200};
  auto range_max = vec_2d<T>{200, 200};

  auto d_points = rmm::device_uvector<vec_2d<T>>(num_points, rmm::cuda_stream_default);
  generate_points(d_points.begin(), d_points.end(), range_min, range_max);

  auto d_x = rmm::device_uvector<T>(num_points, rmm::cuda_stream_default);
  auto d_y = rmm::device_uvector<T>(num_points, rmm::cuda_stream_default);

  thrust::transform(
    rmm::exec_policy(), d_points.begin(), d_points.end(), d_x.begin(), [] __device__(auto point) {
      return point.x;
    });
  thrust::transform(
    rmm::exec_policy(), d_points.begin(), d_points.end(), d_y.begin(), [] __device__(auto point) {
      return point.y;
    });

  auto xs = cudf::column(cudf::data_type{cudf::type_to_id<T>()}, num_points, d_x.release());
  auto ys = cudf::column(cudf::data_type{cudf::type_to_id<T>()}, num_points, d_y.release());

  state.add_element_count(num_points);

  points_in_spatial_window(window_min.x, window_max.x, window_min.y, window_max.y, xs, ys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto points_in =
      points_in_spatial_window(window_min.x, window_max.x, window_min.y, window_max.y, xs, ys);
  });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(points_in_spatial_window_benchmark, NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("NumPoints", {100'000, 1'000'000, 10'000'000, 100'000'000});
