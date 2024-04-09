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
#include <cuspatial/range/multipoint_range.cuh>
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
template <collection_type_id lhs_is_multipoint, collection_type_id rhs_is_multipoint>
struct pairwise_point_distance_impl {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    geometry_column_view const& multipoints1,
    geometry_column_view const& multipoints2,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    auto size = multipoints1.size();

    auto distances = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()}, size, cudf::mask_state::UNALLOCATED, stream, mr);

    auto lhs = make_multipoint_range<lhs_is_multipoint, T, cudf::size_type>(multipoints1);
    auto rhs = make_multipoint_range<rhs_is_multipoint, T, cudf::size_type>(multipoints2);

    pairwise_point_distance(lhs, rhs, distances->mutable_view().begin<T>(), stream);

    return distances;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Point distances only supports floating point coordinates.");
  }
};

template <collection_type_id lhs_is_multipoint, collection_type_id rhs_is_multipoint>
struct pairwise_point_distance_functor {
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multipoints1,
                                           geometry_column_view const& multipoints2,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    CUSPATIAL_EXPECTS(multipoints1.geometry_type() == geometry_type_id::POINT &&
                        multipoints2.geometry_type() == geometry_type_id::POINT,
                      "Unexpected input geometry types.");

    CUSPATIAL_EXPECTS(multipoints1.coordinate_type() == multipoints2.coordinate_type(),
                      "Input coordinates must have the same floating point type.");

    CUSPATIAL_EXPECTS(multipoints1.size() == multipoints2.size(),
                      "Inputs should have the same number of geometries.");

    return cudf::type_dispatcher(
      multipoints1.coordinate_type(),
      pairwise_point_distance_impl<lhs_is_multipoint, rhs_is_multipoint>{},
      multipoints1,
      multipoints2,
      stream,
      mr);
  }
};

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_point_distance(geometry_column_view const& multipoints1,
                                                      geometry_column_view const& multipoints2,
                                                      rmm::device_async_resource_ref mr)
{
  return multi_geometry_double_dispatch<detail::pairwise_point_distance_functor>(
    multipoints1.collection_type(),
    multipoints2.collection_type(),
    multipoints1,
    multipoints2,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
