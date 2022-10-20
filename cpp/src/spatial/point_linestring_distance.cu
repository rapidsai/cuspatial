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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/point_linestring_distance.cuh>
#include <cuspatial/experimental/ranges/multilinestring_range.cuh>
#include <cuspatial/experimental/ranges/multipoint_range.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <memory>
#include <type_traits>

#include "../utility/double_boolean_dispatch.hpp"
#include "../utility/iterator.hpp"

namespace cuspatial {

namespace detail {

namespace {

template <bool is_multi_point, bool is_multi_linestring>
struct pairwise_point_linestring_distance_impl {
  using SizeType = cudf::device_span<cudf::size_type const>::size_type;

  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(
    SizeType num_pairs,
    std::optional<cudf::device_span<cudf::size_type const>> multipoint_geometry_offsets,
    cudf::column_view const& points_xy,
    std::optional<cudf::device_span<cudf::size_type const>> multilinestring_geometry_offsets,
    cudf::device_span<cudf::size_type const> linestring_part_offsets,
    cudf::column_view const& linestring_points_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto const num_points            = static_cast<SizeType>(points_xy.size() / 2);
    auto const num_linestring_points = static_cast<SizeType>(linestring_points_xy.size() / 2);
    auto const num_linestring_parts  = static_cast<SizeType>(linestring_part_offsets.size() - 1);

    auto output = cudf::make_numeric_column(
      points_xy.type(), num_pairs, cudf::mask_state::UNALLOCATED, stream, mr);

    auto point_geometry_it_first =
      get_geometry_iterator_functor<is_multi_point>{}(multipoint_geometry_offsets);
    auto points_it = make_vec_2d_iterator(points_xy.begin<T>());

    auto linestring_geometry_it_first =
      get_geometry_iterator_functor<is_multi_linestring>{}(multilinestring_geometry_offsets);
    auto linestring_points_it = make_vec_2d_iterator(linestring_points_xy.begin<T>());

    auto output_begin = output->mutable_view().begin<T>();

    auto multipoints =
      make_multipoint_range(num_pairs, point_geometry_it_first, num_points, points_it);

    auto multilinestrings = make_multilinestring_range(num_pairs,
                                                       linestring_geometry_it_first,
                                                       num_linestring_parts,
                                                       linestring_part_offsets.begin(),
                                                       num_linestring_points,
                                                       linestring_points_it);

    cuspatial::pairwise_point_linestring_distance(
      multipoints, multilinestrings, output_begin, stream);

    return output;
  }

  template <typename T, CUDF_ENABLE_IF(!std::is_floating_point_v<T>), typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)

  {
    CUSPATIAL_FAIL("Point-linestring distance API only supports floating point coordinates.");
  }
};

}  // namespace

template <bool is_multi_point, bool is_multi_linestring>
struct pairwise_point_linestring_distance_functor {
  std::unique_ptr<cudf::column> operator()(
    std::optional<cudf::device_span<cudf::size_type const>> multipoint_geometry_offsets,
    cudf::column_view const& points_xy,
    std::optional<cudf::device_span<cudf::size_type const>> multilinestring_geometry_offsets,
    cudf::device_span<cudf::size_type const> linestring_part_offsets,
    cudf::column_view const& linestring_points_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    CUSPATIAL_EXPECTS(points_xy.size() % 2 == 0 && linestring_points_xy.size() % 2 == 0,
                      "Points array must contain even number of coordinates.");

    auto num_lhs =
      is_multi_point ? multipoint_geometry_offsets.value().size() : (points_xy.size() / 2 + 1);
    auto num_rhs = is_multi_linestring ? multilinestring_geometry_offsets.value().size()
                                       : linestring_part_offsets.size();

    CUSPATIAL_EXPECTS(num_lhs == num_rhs, "Mismatch number of points and linestrings.");

    CUSPATIAL_EXPECTS(points_xy.type() == linestring_points_xy.type(),
                      "Points and linestring coordinates must have the same type.");

    CUSPATIAL_EXPECTS(!(points_xy.has_nulls() || linestring_points_xy.has_nulls()),
                      "All inputs must not have nulls.");

    if (num_rhs - 1 == 0) return cudf::make_empty_column(points_xy.type());

    return cudf::type_dispatcher(
      points_xy.type(),
      pairwise_point_linestring_distance_impl<is_multi_point, is_multi_linestring>{},
      num_lhs - 1,
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

std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  std::optional<cudf::device_span<cudf::size_type const>> multipoint_geometry_offsets,
  cudf::column_view const& points_xy,
  std::optional<cudf::device_span<cudf::size_type const>> multilinestring_geometry_offsets,
  cudf::device_span<cudf::size_type const> linestring_part_offsets,
  cudf::column_view const& linestring_points_xy,
  rmm::mr::device_memory_resource* mr)
{
  return double_boolean_dispatch<detail::pairwise_point_linestring_distance_functor>(
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
