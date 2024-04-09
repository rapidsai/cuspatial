/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuspatial/error.hpp>
#include <cuspatial/pairwise_multipoint_equals_count.cuh>
#include <cuspatial/range/multipoint_range.cuh>
#include <cuspatial/types.hpp>

#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <type_traits>
#include <utility>

namespace cuspatial {
namespace detail {
namespace {

template <collection_type_id is_multi_point_lhs, collection_type_id is_multi_point_rhs>
struct pairwise_multipoint_equals_count_impl {
  using SizeType = cudf::device_span<cudf::size_type const>::size_type;

  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& lhs,
                                           geometry_column_view const& rhs,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto size = lhs.size();  // lhs is a buffer of xy coords
    auto type = cudf::data_type(cudf::type_to_id<uint32_t>());
    auto result =
      cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);

    auto lhs_range = make_multipoint_range<is_multi_point_lhs, T, cudf::size_type>(lhs);
    auto rhs_range = make_multipoint_range<is_multi_point_rhs, T, cudf::size_type>(rhs);

    cuspatial::pairwise_multipoint_equals_count(
      lhs_range, rhs_range, result->mutable_view().begin<uint32_t>(), stream);

    return result;
  }

  template <typename T, CUDF_ENABLE_IF(!std::is_floating_point_v<T>), typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)

  {
    CUSPATIAL_FAIL("pairwise_multipoint_equals_count only supports floating point types.");
  }
};

}  // namespace

template <collection_type_id is_multi_point_lhs, collection_type_id is_multi_point_rhs>
struct pairwise_multipoint_equals_count {
  std::unique_ptr<cudf::column> operator()(geometry_column_view lhs,
                                           geometry_column_view rhs,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    return cudf::type_dispatcher(
      lhs.coordinate_type(),
      pairwise_multipoint_equals_count_impl<is_multi_point_lhs, is_multi_point_rhs>{},
      lhs,
      rhs,
      stream,
      mr);
  }
};

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_multipoint_equals_count(geometry_column_view const& lhs,
                                                               geometry_column_view const& rhs,
                                                               rmm::device_async_resource_ref mr)
{
  CUSPATIAL_EXPECTS(lhs.geometry_type() == geometry_type_id::POINT &&
                      rhs.geometry_type() == geometry_type_id::POINT,

                    "pairwise_multipoint_equals_count only supports POINT geometries"
                    "for both lhs and rhs");

  CUSPATIAL_EXPECTS(lhs.coordinate_type() == rhs.coordinate_type(),
                    "Input geometries must have the same coordinate data types.");

  CUSPATIAL_EXPECTS(lhs.size() == rhs.size(),
                    "Input geometries must have the same number of multipoints.");

  return multi_geometry_double_dispatch<detail::pairwise_multipoint_equals_count>(
    lhs.collection_type(), rhs.collection_type(), lhs, rhs, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
