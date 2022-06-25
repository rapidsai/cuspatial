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
#include <nvbench/nvbench.cuh>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/linestring_distance.cuh>
#include <cuspatial/experimental/type_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <memory>

namespace cuspatial {

/**
 * @brief Helper to generate linestrings used for benchmarks.
 *
 * The generator adopts a walking algorithm. The ith point is computed by
 * walking (cos(i) * segment_length, sin(i) * segment_length) from the `i-1`
 * point. The initial point of the linestring is at `(init_xy, init_xy)`.
 * Since equidistance sampling on a sinusoid will result in random values,
 * the shape of the linestring is random.
 *
 * The number of line segments per linestring is constrolled by
 * `num_segment_per_string`.
 *
 * Since the outreach upper bound of the linestring group is
 * `(init_xy + num_strings * num_segments_per_string * segment_length)`,
 * user may control the locality of the linestring group via these four
 * arguments. It's important to control the locality between pairs of
 * the linestrings. Linestrings pair that do not intersect will take
 * the longest compute path in the kernel and will benchmark the worst
 * case performance of the API.
 *
 * @tparam T The floating point type for the coordinates
 * @param num_strings Total number of linestrings
 * @param num_segments_per_string Number of line segments per linestring
 * @param segment_length Length of each segment, or stride of walk
 * @param init_xy The initial coordinate to start the walk
 * @param stream The CUDA stream to use for device memory operations and kernel launches
 * @return A tuple of x and y coordinates of points and offsets to which the first point
 * of each linestring starts.
 *
 */
template <typename T>
std::tuple<rmm::device_vector<T>, rmm::device_vector<T>, rmm::device_vector<int32_t>>
generate_linestring(int32_t num_strings,
                    int32_t num_segments_per_string,
                    T segment_length,
                    T init_xy,
                    rmm::cuda_stream_view stream)
{
  int32_t num_points = num_strings * (num_segments_per_string + 1);

  auto offset_iter = detail::make_counting_transform_iterator(
    0, [num_segments_per_string] __device__(auto i) { return i * num_segments_per_string; });
  auto points_x_iter =
    detail::make_counting_transform_iterator(0, [] __device__(auto i) { return cos(i); });
  auto points_y_iter =
    detail::make_counting_transform_iterator(0, [] __device__(auto i) { return sin(i); });

  rmm::device_vector<int32_t> offsets(offset_iter, offset_iter + num_strings);
  rmm::device_vector<T> points_x(points_x_iter, points_x_iter + num_points);
  rmm::device_vector<T> points_y(points_y_iter, points_y_iter + num_points);

  auto random_walk_func = [segment_length] __device__(T prev, T rad) {
    return prev + segment_length * rad;
  };
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         points_x.begin(),
                         points_x.end(),
                         points_x.begin(),
                         init_xy,
                         random_walk_func);

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         points_y.begin(),
                         points_y.end(),
                         points_y.begin(),
                         init_xy,
                         random_walk_func);

  return std::tuple(std::move(points_x), std::move(points_y), std::move(offsets));
}

template <typename T>
void pairwise_linestring_distance_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;

  auto const num_string_pairs{state.get_int64("NumStrings")},
    num_segments_per_string{state.get_int64("NumSegmentsPerString")};
  auto stream = rmm::cuda_stream_default;

  auto [ls1_x, ls1_y, ls1_offset] =
    generate_linestring<T>(num_string_pairs, num_segments_per_string, 1, 0, stream);
  auto [ls2_x, ls2_y, ls2_offset] =
    generate_linestring<T>(num_string_pairs, num_segments_per_string, 1, 100, stream);

  auto ls1_offset_begin = ls1_offset.begin();
  auto ls2_offset_begin = ls2_offset.begin();
  auto ls1_points_begin = cuspatial::make_cartesian_2d_iterator(ls1_x.begin(), ls1_y.begin());
  auto ls2_points_begin = cuspatial::make_cartesian_2d_iterator(ls2_x.begin(), ls2_y.begin());
  auto distances        = rmm::device_vector<T>(ls1_x.size());
  auto out_it           = distances.begin();

  cudaStreamSynchronize(stream.value());

  auto const total_points = ls1_x.size() + ls2_x.size();

  state.add_element_count(num_string_pairs, "LineStringPairs");
  state.add_element_count(total_points, "NumPoints");
  state.add_global_memory_reads<T>(total_points * 2, "CoordinatesDataSize");
  state.add_global_memory_reads<int32_t>(num_string_pairs * 2, "OffsetsDataSize");
  state.add_global_memory_writes<T>(num_string_pairs);

  state.exec(nvbench::exec_tag::sync,
             [&ls1_offset_begin,
              &num_string_pairs,
              &ls1_points_begin,
              ls1_size = ls1_x.size(),
              &ls2_offset_begin,
              &ls2_points_begin,
              ls2_size = ls2_x.size(),
              &out_it](nvbench::launch& launch) {
               cuspatial::pairwise_linestring_distance(ls1_offset_begin,
                                                       ls1_offset_begin + num_string_pairs,
                                                       ls1_points_begin,
                                                       ls1_points_begin + ls1_size,
                                                       ls2_offset_begin,
                                                       ls2_points_begin,
                                                       ls2_points_begin + ls2_size,
                                                       out_it);
             });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(pairwise_linestring_distance_benchmark, NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("NumStrings", {1'000, 10'000, 100'000})
  .add_int64_axis("NumSegmentsPerString", {10, 100, 1'000});

}  // namespace cuspatial
