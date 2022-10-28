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
#include <cuspatial/experimental/point_distance.cuh>
#include <cuspatial/experimental/ranges/multipoint_range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {
template <bool is_multipoint1, bool is_multipoint2>
struct pairwise_point_distance_impl {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::size_type num_pairs,
    std::optional<cudf::device_span<cudf::size_type const>> multipoints1_offset,
    cudf::column_view const& points1_xy,
    std::optional<cudf::device_span<cudf::size_type const>> multipoints2_offset,
    cudf::column_view const& points2_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto distances = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()}, num_pairs, cudf::mask_state::UNALLOCATED, stream, mr);

    auto multipoint1_offset_it =
      get_geometry_iterator_functor<is_multipoint1>{}(multipoints1_offset);
    auto multipoint2_offset_it =
      get_geometry_iterator_functor<is_multipoint2>{}(multipoints2_offset);

    auto points1_it = make_vec_2d_iterator(points1_xy.begin<T>());
    auto points2_it = make_vec_2d_iterator(points2_xy.begin<T>());

    auto multipoint1_its =
      make_multipoint_range(num_pairs, multipoint1_offset_it, points1_xy.size() / 2, points1_it);
    auto multipoint2_its =
      make_multipoint_range(num_pairs, multipoint2_offset_it, points2_xy.size() / 2, points2_it);

    pairwise_point_distance(
      multipoint1_its, multipoint2_its, distances->mutable_view().begin<T>(), stream);

    return distances;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Point distances only supports floating point coordinates.");
  }
};

template <bool is_multipoint1, bool is_multipoint2>
struct pairwise_point_distance_functor {
  std::unique_ptr<cudf::column> operator()(
    std::optional<cudf::device_span<cudf::size_type const>> multipoints1_offset,
    cudf::column_view const& points1_xy,
    std::optional<cudf::device_span<cudf::size_type const>> multipoints2_offset,
    cudf::column_view const& points2_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    CUSPATIAL_EXPECTS(points1_xy.size() % 2 == 0 and points2_xy.size() % 2 == 0,
                      "Coordinate array should contain even number of points.");
    CUSPATIAL_EXPECTS(points1_xy.type() == points2_xy.type(),
                      "The types of point coordinates arrays mismatch.");
    CUSPATIAL_EXPECTS(not points1_xy.has_nulls() and not points2_xy.has_nulls(),
                      "The coordinate columns cannot have nulls.");

    auto num_lhs = is_multipoint1 ? multipoints1_offset.value().size() - 1 : points1_xy.size() / 2;
    auto num_rhs = is_multipoint2 ? multipoints2_offset.value().size() - 1 : points2_xy.size() / 2;

    CUSPATIAL_EXPECTS(num_lhs == num_rhs, "Mismatch number of (multi)point(s) in input.");

    if (num_lhs == 0) { return cudf::empty_like(points1_xy); }

    return cudf::type_dispatcher(points1_xy.type(),
                                 pairwise_point_distance_impl<is_multipoint1, is_multipoint2>{},
                                 num_lhs,
                                 multipoints1_offset,
                                 points1_xy,
                                 multipoints2_offset,
                                 points2_xy,
                                 stream,
                                 mr);
  }
};

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_point_distance(
  std::optional<cudf::device_span<cudf::size_type const>> multipoints1_offset,
  cudf::column_view const& points1_xy,
  std::optional<cudf::device_span<cudf::size_type const>> multipoints2_offset,
  cudf::column_view const& points2_xy,
  rmm::mr::device_memory_resource* mr)
{
  return double_boolean_dispatch<detail::pairwise_point_distance_functor>(
    multipoints1_offset.has_value(),
    multipoints2_offset.has_value(),
    multipoints1_offset,
    points1_xy,
    multipoints2_offset,
    points2_xy,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
