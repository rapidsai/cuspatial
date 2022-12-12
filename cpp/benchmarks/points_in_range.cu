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
#include <cuspatial_test/random.cuh>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/points_in_range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/tabulate.h>

#include <memory>

using namespace cuspatial;

/**
 * @brief Helper to generate random points within a range
 *
 * @p begin and @p end must be iterators to device-accessible memory
 *
 * @tparam PointsIter The type of the iterator to the output points container
 * @tparam T The floating point type for the coordinates
 * @param begin The start of the sequence of points to generate
 * @param end The end of the sequence of points to generate
 *
 * @param range the lower left range corner
 * @param range the upper right range corner
 *
 */
template <class PointsIter, typename T>
void generate_points(PointsIter begin, PointsIter end, vec_2d<T> range_min, vec_2d<T> range_max)
{
  auto engine_x = deterministic_engine(std::distance(begin, end));
  auto engine_y = deterministic_engine(2 * std::distance(begin, end));

  auto x_dist = make_uniform_dist(range_min.x, range_max.x);
  auto y_dist = make_uniform_dist(range_min.y, range_max.y);

  auto x_gen = value_generator{range_min.x, range_max.x, engine_x, x_dist};
  auto y_gen = value_generator{range_min.y, range_max.y, engine_y, y_dist};

  thrust::tabulate(rmm::exec_policy(), begin, end, [x_gen, y_gen] __device__(size_t n) mutable {
    return vec_2d<T>{x_gen(n), y_gen(n)};
  });
}

template <typename T>
void points_in_range_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;

  auto const num_points{state.get_int64("NumPoints")};

  auto range_min = vec_2d<T>{-100, -100};
  auto range_max = vec_2d<T>{100, 100};

  auto generate_min = vec_2d<T>{-200, -200};
  auto generate_max = vec_2d<T>{200, 200};

  auto points = rmm::device_uvector<cuspatial::vec_2d<T>>(num_points, rmm::cuda_stream_default);

  generate_points(points.begin(), points.end(), generate_min, generate_max);

  CUSPATIAL_CUDA_TRY(cudaDeviceSynchronize());

  state.add_element_count(num_points);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto stream = rmm::cuda_stream_view(launch.get_stream());
    auto num_points_in =
      cuspatial::count_points_in_range(range_min, range_max, points.begin(), points.end(), stream);

    auto result_points = rmm::device_uvector<cuspatial::vec_2d<T>>(num_points_in, stream);

    cuspatial::copy_points_in_range(
      range_min, range_max, points.begin(), points.end(), result_points.begin(), stream);
  });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(points_in_range_benchmark, NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("NumPoints", {100'000, 1'000'000, 10'000'000, 100'000'000});
