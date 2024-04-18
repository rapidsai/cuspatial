/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "utility/multi_geometry_dispatch.hpp"

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/distance.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/range/multilinestring_range.cuh>
#include <cuspatial/range/multipoint_range.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/types.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <type_traits>

namespace cuspatial {

namespace detail {

namespace {

template <collection_type_id is_multi_point, collection_type_id is_multi_linestring>
struct pairwise_point_linestring_distance_impl {
  using SizeType = cudf::device_span<cudf::size_type const>::size_type;

  template <typename T, CUSPATIAL_ENABLE_IF(std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multipoints,
                                           geometry_column_view const& multilinestrings,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto size = multipoints.size();

    auto distances = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()}, size, cudf::mask_state::UNALLOCATED, stream, mr);

    auto lhs = make_multipoint_range<is_multi_point, T, cudf::size_type>(multipoints);
    auto rhs =
      make_multilinestring_range<is_multi_linestring, T, cudf::size_type>(multilinestrings);

    cuspatial::pairwise_point_linestring_distance(
      lhs, rhs, distances->mutable_view().begin<T>(), stream);

    return distances;
  }

  template <typename T, CUSPATIAL_ENABLE_IF(!std::is_floating_point_v<T>), typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Point-linestring distance API only supports floating point coordinates.");
  }
};

}  // namespace

template <collection_type_id is_multi_point, collection_type_id is_multi_linestring>
struct pairwise_point_linestring_distance_functor {
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multipoints,
                                           geometry_column_view const& multilinestrings,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    CUSPATIAL_EXPECTS(multipoints.geometry_type() == geometry_type_id::POINT &&
                        multilinestrings.geometry_type() == geometry_type_id::LINESTRING,
                      "Unexpected input geometry types.");

    CUSPATIAL_EXPECTS(multipoints.coordinate_type() == multilinestrings.coordinate_type(),
                      "Inputs must have the same coordinate type.");

    CUSPATIAL_EXPECTS(multipoints.size() == multilinestrings.size(),
                      "Inputs should have the same number of geometries.");

    return cudf::type_dispatcher(
      multipoints.coordinate_type(),
      pairwise_point_linestring_distance_impl<is_multi_point, is_multi_linestring>{},
      multipoints,
      multilinestrings,
      stream,
      mr);
  }
};

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  geometry_column_view const& multipoints,
  geometry_column_view const& multilinestrings,
  rmm::device_async_resource_ref mr)
{
  return multi_geometry_double_dispatch<detail::pairwise_point_linestring_distance_functor>(
    multipoints.collection_type(),
    multilinestrings.collection_type(),
    multipoints,
    multilinestrings,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
