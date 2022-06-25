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

#include <cuspatial/experimental/linestring_distance.cuh>
#include <cuspatial/experimental/type_utils.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
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
std::tuple<std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>,
           rmm::device_uvector<cudf::size_type>>
generate_linestring(cudf::size_type num_strings,
                    cudf::size_type num_segments_per_string,
                    T segment_length,
                    T init_xy,
                    rmm::cuda_stream_view stream)
{
  cudf::size_type num_points = num_strings * (num_segments_per_string + 1);
  rmm::device_uvector<cudf::size_type> offsets(num_points, stream);
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(static_cast<cudf::size_type>(0)),
    thrust::make_counting_iterator(static_cast<cudf::size_type>(num_points)),
    offsets.begin(),
    [num_segments_per_string] __device__(auto i) { return i * num_segments_per_string; });
  auto points_x = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<T>()}, num_points, cudf::mask_state::UNALLOCATED);
  auto points_y = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<T>()}, num_points, cudf::mask_state::UNALLOCATED);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator(static_cast<cudf::size_type>(0)),
                    thrust::counting_iterator(static_cast<cudf::size_type>(num_points)),
                    points_x->mutable_view().begin<T>(),
                    [] __device__(auto i) { return cos(i); });
  thrust::exclusive_scan(
    rmm::exec_policy(stream),
    points_x->view().begin<T>(),
    points_x->view().end<T>(),
    points_x->mutable_view().begin<T>(),
    init_xy,
    [segment_length] __device__(T prev, T rad) { return prev + segment_length * rad; });
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator(static_cast<cudf::size_type>(0)),
                    thrust::counting_iterator(static_cast<cudf::size_type>(num_points)),
                    points_y->mutable_view().begin<T>(),
                    [] __device__(auto i) { return sin(i); });
  thrust::exclusive_scan(
    rmm::exec_policy(stream),
    points_y->view().begin<T>(),
    points_y->view().end<T>(),
    points_y->mutable_view().begin<T>(),
    init_xy,
    [segment_length] __device__(T prev, T rad) { return prev + segment_length * rad; });

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

  auto ls1_points_begin = cuspatial::make_cartesian_2d_iterator(ls1_x->view().template begin<T>(),
                                                                ls1_y->view().template begin<T>());
  auto ls2_points_begin = cuspatial::make_cartesian_2d_iterator(ls2_x->view().template begin<T>(),
                                                                ls2_y->view().template begin<T>());
  auto distances        = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<T>()}, ls1_x->size(), cudf::mask_state::UNALLOCATED);
  auto out_it = distances->mutable_view().template begin<T>();

  cudaStreamSynchronize(stream.value());

  auto const total_points = ls1_x->size() + ls2_x->size();

  state.add_element_count(num_string_pairs, "LineStringPairs");
  state.add_element_count(total_points, "NumPoints");
  state.add_global_memory_reads<T>(total_points * 2, "CoordinatesDataSize");
  state.add_global_memory_reads<cudf::size_type>(num_string_pairs * 2, "OffsetsDataSize");
  state.add_global_memory_writes<T>(num_string_pairs);

  state.exec(nvbench::exec_tag::sync,
             [ls1_offset = cudf::device_span<cudf::size_type>(ls1_offset),
              &ls1_points_begin,
              ls1_size   = ls1_x->size(),
              ls2_offset = cudf::device_span<cudf::size_type>(ls2_offset),
              &ls2_points_begin,
              ls2_size = ls2_x->size(),
              &out_it](nvbench::launch& launch) {
               cuspatial::pairwise_linestring_distance(ls1_offset.begin(),
                                                       ls1_offset.end(),
                                                       ls1_points_begin,
                                                       ls1_points_begin + ls1_size,
                                                       ls2_offset.begin(),
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
