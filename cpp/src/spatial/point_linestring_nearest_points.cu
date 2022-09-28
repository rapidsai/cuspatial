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

#include "../utility/double_boolean_dispatch.hpp"
#include "../utility/iterator.hpp"

#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/point_linestring_nearest_points.cuh>
#include <cuspatial/point_linestring_nearest_points.hpp>
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
#include <utility>

namespace cuspatial {

namespace detail {

namespace {

template <bool is_multi_point, bool is_multi_linestring>
struct pairwise_point_linestring_nearest_points_impl {
  using SizeType = cudf::device_span<cudf::size_type>::size_type;

  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  point_linestring_nearest_points_result operator()(
    cudf::size_type num_pairs,
    std::optional<cudf::device_span<cudf::size_type const>> multipoint_geometry_offsets,
    cudf::column_view points_xy,
    std::optional<cudf::device_span<cudf::size_type const>> multilinestring_geometry_offsets,
    cudf::device_span<cudf::size_type const> linestring_offsets,
    cudf::column_view linestring_points_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto num_points            = static_cast<SizeType>(points_xy.size() / 2);
    auto num_linestring_points = static_cast<SizeType>(linestring_points_xy.size() / 2);

    auto point_geometry_it =
      get_geometry_iterator_functor<is_multi_point>{}(multipoint_geometry_offsets);
    auto points_it = make_vec_2d_iterator(points_xy.begin<T>());

    auto linestring_geometry_it =
      get_geometry_iterator_functor<is_multi_linestring>{}(multilinestring_geometry_offsets);
    auto linestring_points_it = make_vec_2d_iterator(linestring_points_xy.begin<T>());

    auto segment_idx =
      cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                num_pairs,
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);

    auto nearest_points_xy = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<T>()},
                                                       num_pairs * 2,
                                                       cudf::mask_state::UNALLOCATED,
                                                       stream,
                                                       mr);
    auto nearest_points_it =
      make_vec_2d_output_iterator(nearest_points_xy->mutable_view().begin<T>());

    if constexpr (!is_multi_point && !is_multi_linestring) {
      auto output_its = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_discard_iterator(),
                           thrust::make_discard_iterator(),
                           segment_idx->mutable_view().begin<cudf::size_type>(),
                           nearest_points_it));

      pairwise_point_linestring_nearest_points(point_geometry_it,
                                               point_geometry_it + num_pairs + 1,
                                               points_it,
                                               points_it + num_points,
                                               linestring_geometry_it,
                                               linestring_offsets.begin(),
                                               linestring_offsets.end(),
                                               linestring_points_it,
                                               linestring_points_it + num_linestring_points,
                                               output_its,
                                               stream);

      return point_linestring_nearest_points_result{
        std::nullopt, std::nullopt, std::move(segment_idx), std::move(nearest_points_xy)};
    } else if constexpr (is_multi_point && !is_multi_linestring) {
      auto nearest_point_idx =
        cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                  num_pairs,
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  mr);
      auto output_its = thrust::make_zip_iterator(
        thrust::make_tuple(nearest_point_idx->mutable_view().begin<cudf::size_type>(),
                           thrust::make_discard_iterator(),
                           segment_idx->mutable_view().begin<cudf::size_type>(),
                           nearest_points_it));

      pairwise_point_linestring_nearest_points(point_geometry_it,
                                               point_geometry_it + num_pairs + 1,
                                               points_it,
                                               points_it + num_points,
                                               linestring_geometry_it,
                                               linestring_offsets.begin(),
                                               linestring_offsets.end(),
                                               linestring_points_it,
                                               linestring_points_it + num_linestring_points,
                                               output_its,
                                               stream);

      return point_linestring_nearest_points_result{std::move(nearest_point_idx),
                                                    std::nullopt,
                                                    std::move(segment_idx),
                                                    std::move(nearest_points_xy)};
    } else if constexpr (!is_multi_point && is_multi_linestring) {
      auto nearest_linestring_idx =
        cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                  num_pairs,
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  mr);
      auto output_its = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_discard_iterator(),
                           nearest_linestring_idx->mutable_view().begin<cudf::size_type>(),
                           segment_idx->mutable_view().begin<cudf::size_type>(),
                           nearest_points_it));

      pairwise_point_linestring_nearest_points(point_geometry_it,
                                               point_geometry_it + num_pairs + 1,
                                               points_it,
                                               points_it + num_points,
                                               linestring_geometry_it,
                                               linestring_offsets.begin(),
                                               linestring_offsets.end(),
                                               linestring_points_it,
                                               linestring_points_it + num_linestring_points,
                                               output_its,
                                               stream);

      return point_linestring_nearest_points_result{std::nullopt,
                                                    std::move(nearest_linestring_idx),
                                                    std::move(segment_idx),
                                                    std::move(nearest_points_xy)};
    } else {
      auto nearest_point_idx =
        cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                  num_pairs,
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  mr);
      auto nearest_linestring_idx =
        cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                  num_pairs,
                                  cudf::mask_state::UNALLOCATED,
                                  stream,
                                  mr);
      auto output_its = thrust::make_zip_iterator(
        thrust::make_tuple(nearest_point_idx->mutable_view().begin<cudf::size_type>(),
                           nearest_linestring_idx->mutable_view().begin<cudf::size_type>(),
                           segment_idx->mutable_view().begin<cudf::size_type>(),
                           nearest_points_it));

      pairwise_point_linestring_nearest_points(point_geometry_it,
                                               point_geometry_it + num_pairs + 1,
                                               points_it,
                                               points_it + num_points,
                                               linestring_geometry_it,
                                               linestring_offsets.begin(),
                                               linestring_offsets.end(),
                                               linestring_points_it,
                                               linestring_points_it + num_linestring_points,
                                               output_its,
                                               stream);

      return point_linestring_nearest_points_result{std::move(nearest_point_idx),
                                                    std::move(nearest_linestring_idx),
                                                    std::move(segment_idx),
                                                    std::move(nearest_points_xy)};
    }
  }

  template <typename T, CUDF_ENABLE_IF(!std::is_floating_point_v<T>), typename... Args>
  point_linestring_nearest_points_result operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Input coordinates for nearest point must be floating point types.");
  }
};

}  // namespace

template <bool is_multi_point, bool is_multi_linestring>
struct pairwise_point_linestring_nearest_points_functor {
  point_linestring_nearest_points_result operator()(
    std::optional<cudf::device_span<cudf::size_type const>> multipoint_geometry_offsets,
    cudf::column_view points_xy,
    std::optional<cudf::device_span<cudf::size_type const>> multilinestring_geometry_offsets,
    cudf::device_span<cudf::size_type const> linestring_part_offsets,
    cudf::column_view linestring_points_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    CUSPATIAL_EXPECTS(points_xy.size() % 2 == 0 && linestring_points_xy.size() % 2 == 0,
                      "Points array must contain even number of coordinates.");

    auto num_lhs =
      is_multi_point ? multipoint_geometry_offsets.value().size() : (points_xy.size() / 2 + 1);
    auto num_rhs = is_multi_linestring ? multilinestring_geometry_offsets.value().size()
                                       : linestring_part_offsets.size();

    CUSPATIAL_EXPECTS(num_lhs == num_rhs,
                      "Mismatch number of (multi)points and (multi)linestrings.");
    CUSPATIAL_EXPECTS(points_xy.type() == linestring_points_xy.type(),
                      "Points and linestring coordinates must have the same type.");
    CUSPATIAL_EXPECTS(!(points_xy.has_nulls() || linestring_points_xy.has_nulls()),
                      "All inputs must not have nulls.");

    return cudf::type_dispatcher(
      points_xy.type(),
      pairwise_point_linestring_nearest_points_impl<is_multi_point, is_multi_linestring>{},
      num_rhs - 1,
      multipoint_geometry_offsets,
      points_xy,
      multilinestring_geometry_offsets,
      linestring_part_offsets,
      linestring_points_xy,
      stream,
      mr);
  }
};

}  // namespace detail

point_linestring_nearest_points_result pairwise_point_linestring_nearest_points(
  std::optional<cudf::device_span<cudf::size_type const>> multipoint_geometry_offsets,
  cudf::column_view points_xy,
  std::optional<cudf::device_span<cudf::size_type const>> multilinestring_geometry_offsets,
  cudf::device_span<cudf::size_type const> linestring_part_offsets,
  cudf::column_view linestring_points_xy,
  rmm::mr::device_memory_resource* mr)
{
  return double_boolean_dispatch<detail::pairwise_point_linestring_nearest_points_functor>(
    multipoint_geometry_offsets.has_value(),
    multilinestring_geometry_offsets.has_value(),
    multipoint_geometry_offsets,
    points_xy,
    multilinestring_geometry_offsets,
    linestring_part_offsets,
    linestring_points_xy,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
