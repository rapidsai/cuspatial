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

#include "../utility/multi_geometry_dispatch.hpp"

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/distance.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/range/multilinestring_range.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/types.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <type_traits>
namespace cuspatial {
namespace detail {

template <collection_type_id lhs_is_multilinestring, collection_type_id rhs_is_multilinestring>
struct pairwise_linestring_distance_launch {
  using SizeType = cudf::device_span<cudf::size_type const>::size_type;

  template <typename T, CUSPATIAL_ENABLE_IF(std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multilinestrings1,
                                           geometry_column_view const& multilinestrings2,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto size = multilinestrings1.size();

    auto distances = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()}, size, cudf::mask_state::UNALLOCATED, stream, mr);

    auto lhs =
      make_multilinestring_range<lhs_is_multilinestring, T, cudf::size_type>(multilinestrings1);
    auto rhs =
      make_multilinestring_range<rhs_is_multilinestring, T, cudf::size_type>(multilinestrings2);

    pairwise_linestring_distance(lhs, rhs, distances->mutable_view().begin<T>(), stream);

    return distances;
  }

  template <typename T, CUSPATIAL_ENABLE_IF(!std::is_floating_point_v<T>), typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Linestring distance API only supports floating point coordinates.");
  }
};

template <collection_type_id lhs_is_multilinestring, collection_type_id rhs_is_multilinestring>
struct pairwise_linestring_distance_functor {
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multilinestrings1,
                                           geometry_column_view const& multilinestrings2,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    CUSPATIAL_EXPECTS(multilinestrings1.geometry_type() == geometry_type_id::LINESTRING &&
                        multilinestrings2.geometry_type() == geometry_type_id::LINESTRING,
                      "Unexpected input geometry types.");

    CUSPATIAL_EXPECTS(multilinestrings1.coordinate_type() == multilinestrings2.coordinate_type(),
                      "Inputs must have the same coordinate type.");

    CUSPATIAL_EXPECTS(multilinestrings1.size() == multilinestrings2.size(),
                      "Inputs should have the same number of geometries.");

    return cudf::type_dispatcher(
      multilinestrings1.coordinate_type(),
      pairwise_linestring_distance_launch<lhs_is_multilinestring, rhs_is_multilinestring>{},
      multilinestrings1,
      multilinestrings2,
      stream,
      mr);
  }
};
}  // namespace detail
std::unique_ptr<cudf::column> pairwise_linestring_distance(
  geometry_column_view const& multilinestrings1,
  geometry_column_view const& multilinestrings2,
  rmm::device_async_resource_ref mr)
{
  return multi_geometry_double_dispatch<detail::pairwise_linestring_distance_functor>(
    multilinestrings1.collection_type(),
    multilinestrings2.collection_type(),
    multilinestrings1,
    multilinestrings2,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
