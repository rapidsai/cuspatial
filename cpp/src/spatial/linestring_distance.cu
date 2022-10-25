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

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/linestring_distance.cuh>
#include <cuspatial/experimental/ranges/multilinestring_range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <limits>
#include <memory>
#include <type_traits>
namespace cuspatial {
namespace detail {

template <bool first_is_multilinestring, bool second_is_multilinestring>
struct pairwise_linestring_distance_launch {
  using SizeType = cudf::device_span<cudf::size_type const>::size_type;

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    SizeType num_pairs,
    std::optional<cudf::device_span<cudf::size_type const>> multilinestring1_geometry_offsets,
    cudf::device_span<cudf::size_type const> linestring1_part_offsets,
    cudf::column_view const& linestring1_points_xy,
    std::optional<cudf::device_span<cudf::size_type const>> multilinestring2_geometry_offsets,
    cudf::device_span<cudf::size_type const> linestring2_part_offsets,
    cudf::column_view const& linestring2_points_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto const num_multilinestring1_parts =
      static_cast<SizeType>(linestring1_part_offsets.size() - 1);
    auto const num_multilinestring2_parts =
      static_cast<SizeType>(linestring2_part_offsets.size() - 1);
    auto const num_multilinestring1_points =
      static_cast<SizeType>(linestring1_points_xy.size() / 2);
    auto const num_multilinestring2_points =
      static_cast<SizeType>(linestring2_points_xy.size() / 2);

    auto distances = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()}, num_pairs, cudf::mask_state::UNALLOCATED, stream, mr);

    auto linestring1_coords_it = make_vec_2d_iterator(linestring1_points_xy.begin<T>());
    auto linestring2_coords_it = make_vec_2d_iterator(linestring2_points_xy.begin<T>());

    auto multilinestrings1 = make_multilinestring_range(
      num_pairs,
      get_geometry_iterator_functor<first_is_multilinestring>{}(multilinestring1_geometry_offsets),
      num_multilinestring1_parts,
      linestring1_part_offsets.begin(),
      num_multilinestring1_points,
      linestring1_coords_it);

    auto multilinestrings2 = make_multilinestring_range(
      num_pairs,
      get_geometry_iterator_functor<second_is_multilinestring>{}(multilinestring2_geometry_offsets),
      num_multilinestring2_parts,
      linestring2_part_offsets.begin(),
      num_multilinestring2_points,
      linestring2_coords_it);

    pairwise_linestring_distance(
      multilinestrings1, multilinestrings2, distances->mutable_view().begin<T>(), stream);

    return distances;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Linestring distance API only supports floating point coordinates.");
  }
};

template <bool first_is_multilinestring, bool second_is_multilinestring>
struct pairwise_linestring_distance_functor {
  std::unique_ptr<cudf::column> operator()(
    std::optional<cudf::device_span<cudf::size_type const>> multilinestring1_geometry_offsets,
    cudf::device_span<cudf::size_type const> linestring1_part_offsets,
    cudf::column_view const& linestring1_points_xy,
    std::optional<cudf::device_span<cudf::size_type const>> multilinestring2_geometry_offsets,
    cudf::device_span<cudf::size_type const> linestring2_part_offsets,
    cudf::column_view const& linestring2_points_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    CUSPATIAL_EXPECTS(
      linestring1_points_xy.size() % 2 == 0 && linestring2_points_xy.size() % 2 == 0,
      "Points array must contain even number of coordinates.");

    auto num_lhs = first_is_multilinestring ? multilinestring1_geometry_offsets.value().size() - 1
                                            : linestring1_part_offsets.size() - 1;
    auto num_rhs = second_is_multilinestring ? multilinestring2_geometry_offsets.value().size() - 1
                                             : linestring2_part_offsets.size() - 1;

    CUSPATIAL_EXPECTS(num_lhs == num_rhs, "Mismatch number of points and linestrings.");

    CUSPATIAL_EXPECTS(linestring1_points_xy.type() == linestring2_points_xy.type(),
                      "The types of linestring coordinates arrays mismatch.");

    CUSPATIAL_EXPECTS(!(linestring1_points_xy.has_nulls() || linestring2_points_xy.has_nulls()),
                      "All inputs must not have nulls.");

    if (num_lhs == 0) { return cudf::empty_like(linestring1_points_xy); }

    return cudf::type_dispatcher(
      linestring1_points_xy.type(),
      pairwise_linestring_distance_launch<first_is_multilinestring, second_is_multilinestring>{},
      num_lhs,
      multilinestring1_geometry_offsets,
      linestring1_part_offsets,
      linestring1_points_xy,
      multilinestring2_geometry_offsets,
      linestring2_part_offsets,
      linestring2_points_xy,
      stream,
      mr);
  }
};
}  // namespace detail
std::unique_ptr<cudf::column> pairwise_linestring_distance(
  std::optional<cudf::device_span<cudf::size_type const>> multilinestring1_geometry_offsets,
  cudf::device_span<cudf::size_type const> linestring1_part_offsets,
  cudf::column_view const& linestring1_points_xy,
  std::optional<cudf::device_span<cudf::size_type const>> multilinestring2_geometry_offsets,
  cudf::device_span<cudf::size_type const> linestring2_part_offsets,
  cudf::column_view const& linestring2_points_xy,
  rmm::mr::device_memory_resource* mr)
{
  return double_boolean_dispatch<detail::pairwise_linestring_distance_functor>(
    multilinestring1_geometry_offsets.has_value(),
    multilinestring2_geometry_offsets.has_value(),
    multilinestring1_geometry_offsets,
    linestring1_part_offsets,
    linestring1_points_xy,
    multilinestring2_geometry_offsets,
    linestring2_part_offsets,
    linestring2_points_xy,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
