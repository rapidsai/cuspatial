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

#pragma once

#include <cuspatial/experimental/point_linestring_nearest_point.cuh>
#include <cuspatial/experimental/type_utils.hpp>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reshape.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <optional>

namespace cuspatial {

namespace detail {

namespace {

template <bool has_value>
auto get_part_iterator(std::optional<cudf::device_span<cudf::size_type>>);

template <>
auto get_part_iterator<true>(std::optional<cudf::device_span<cudf::size_type>> opt)
{
  return opt.value().begin();
}

template <>
auto get_part_iterator<false>(std::optional<cudf::device_span<cudf::size_type>>)
{
  return thrust::make_counting_iterator(0);
}

template <bool is_multipoint, bool is_multilinestring>
struct launch_functor {
  using SizeType = cudf::device_span<cudf::size_type>::size_type;

  template <typename T>
  std::tuple<std::optional<std::unique_ptr<cudf::column>>,
             std::unique_ptr<cudf::column>,
             std::unique_ptr<cudf::column>>
  operator()(std::optional<cudf::device_span<cudf::size_type>> multipoint_parts_offsets,
             cudf::column_view points_xy,
             std::optional<cudf::device_span<cudf::size_type>> multilinestring_parts_offsets,
             cudf::device_span<cudf::size_type> linestring_offsets,
             cudf::column_view linestring_points_xy,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    auto num_pairs = is_multipoint ? multipoint_parts_offsets.value().size() : points_xy.size() / 2;
    auto num_points            = static_cast<SizeType>(points_xy.size() / 2);
    auto num_linestring_points = static_cast<SizeType>(linestring_points_xy.size() / 2);

    auto point_parts_it = get_part_iterator<is_multipoint>(multipoint_parts_offsets);
    auto points_it      = interleaved_iterator_to_cartesian_2d_iterator(points_xy.begin<T>());

    auto linestring_parts_it = get_part_iterator<is_multlinestring>(multilinestring_parts_offsets);
    auto linestring_points_it =
      interleaved_iterator_to_cartesian_2d_iterator(linestring_points_xy.begin<T>());

    auto segment_idx = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, num_pairs, cudf::mask_state::UNALLOCATED);

    // Optimize this by creating an interleaving
    auto nearest_points_x = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()}, num_pairs, cudf::mask_state::UNALLOCATED);
    auto nearest_points_y = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()}, num_pairs, cudf::mask_state::UNALLOCATED);

    auto nearest_points_it = make_zipped_cartesian_2d_output_iterator(nearest_points_x.begin<T>(),
                                                                      nearest_points_y.begin<T>());

    if constexpr (is_multilinestring) {
      auto linestring_part_idx = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, num_pairs, cudf::mask_state::UNALLOCATED);
      auto zipped_out = thrust::make_zip_iterator(thrust::make_tuple(
        linestring_part_idx.begin<int32_t>(), segment_idx.begin<int32_t>(), nearest_points_it));
      pairwise_point_linestring_nearest_point(point_parts_it,
                                              point_parts_it + num_pairs,
                                              points_it,
                                              points_it + num_points,
                                              linestring_parts_it,
                                              linestring_parts_it + num_pairs,
                                              linestring_offsets.begin(),
                                              linestring_offsets.end(),
                                              linestring_points_it,
                                              linestring_points_it + num_linestring_points,
                                              zipped_out,
                                              stream);

      auto nearest_points =
        cudf::interleave_columns(cudf::table_view({*nearest_points_x, *nearest_points_y}));
      return std::tuple(
        std::move(linestring_part_idx), std::move(segment_idx), std::move(nearest_points));
    } else {
      auto zipped_out = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_discard_iterator(), segment_idx.begin<int32_t>(), nearest_points_it));
      pairwise_point_linestring_nearest_point(point_parts_it,
                                              point_parts_it + num_pairs,
                                              points_it,
                                              points_it + num_points,
                                              linestring_parts_it,
                                              linestring_parts_it + num_pairs,
                                              linestring_offsets.begin(),
                                              linestring_offsets.end(),
                                              linestring_points_it,
                                              linestring_points_it + num_linestring_points,
                                              zipped_out,
                                              stream);
      auto nearest_points =
        cudf::interleave_columns(cudf::table_view({*nearest_points_x, *nearest_points_y}));
      return std::tuple(std::nullopt, std::move(segment_idx), std::move(nearest_points));
    }
  }
};

}  // namespace

template <template <bool is_multi_1, bool is_multi_2> class Functor,
          typename Optional1,
          typename Optional2,
          typename... Args>
auto optional_double_disptach(Optional1 opt1, Optional2 opt2, Args&&... args)
{
  if (opt1.has_value() && opt2.has_value()) {
    return Functor<true, true>{}(std::forward(args...));
  } else if (!opt1.has_value() && opt2.has_value()) {
    return Functor<false, true>{}(std::forward(args...));
  } else if (opt1.has_value() && !opt2.has_value()) {
    return Functor<true, false>{}(std::forward(args...));
  } else {
    return Functor<false, false>{}(std::forward(args...));
  }
}

template <bool is_multipoint, bool is_multilinestring>
struct pairwise_point_linestring_nearest_point_functor {
  std::tuple<std::optional<std::unique_ptr<cudf::column>>,
             std::unique_ptr<cudf::column>,
             std::unique_ptr<cudf::column>>
  operator()(std::optional<cudf::device_span<cudf::size_type>> multipoint_parts_offsets,
             cudf::column_view points_xy,
             std::optional<cudf::device_span<cudf::size_type>> multilinestring_parts_offsets,
             cudf::device_span<cudf::size_type> linestring_offsets,
             cudf::column_view linestring_points_xy,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    return cudf::type_dispatcher(points_xy.type(),
                                 launch_functor<is_multipoint, is_multilinestring>{},
                                 multipoint_parts_offsets,
                                 points_xy,
                                 multilinestring_parts_offsets,
                                 linestring_offsets,
                                 linestring_points_xy,
                                 stream,
                                 mr);
  }
};

}  // namespace detail

std::tuple<std::optional<std::unique_ptr<cudf::column>>,
           std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>>
pairwise_point_linestring_nearest_point_segment_idx(
  std::optional<cudf::device_span<cudf::size_type>> multipoint_parts_offsets,
  cudf::column_view points_xy,
  std::optional<cudf::device_span<cudf::size_type>> multilinestring_parts_offsets,
  cudf::device_span<cudf::size_type> linestring_offsets,
  cudf::column_view linestring_points_xy,
  rmm::mr::device_memory_resource* mr)
{
  return detail::optional_double_disptach<detail::pairwise_point_linestring_nearest_point_functor>(
    multipoint_parts_offsets,
    multilinestring_parts_offsets,
    multipoint_parts_offsets,
    points_xy,
    multilinestring_parts_offsets,
    linestring_offsets,
    linestring_points_xy,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
