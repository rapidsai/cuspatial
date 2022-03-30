/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuspatial/error.hpp>
#include <cuspatial/polygon_distance.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>
#include <memory>
#include <type_traits>

namespace {

using size_type = cudf::size_type;

/**
 * @brief Device function to compute the distance between polygons
 *
 * The first step is to compute if two polygons intersects or constains
 * one another. If so, the distance is 0. Otherwise, compute every pair
 * of line segment-line segment, point-line segment and point-point for
 * shortest distance.
 *
 */
struct polygon_distance_functor {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::device_span<cudf::size_type> const& poly1_offsets,
    cudf::device_span<cudf::size_type> const& poly1_ring_offsets,
    cudf::column_view const& poly1_xs,
    cudf::column_view const& poly1_ys,
    cudf::device_span<cudf::size_type> const& poly2_offsets,
    cudf::device_span<cudf::size_type> const& poly2_ring_offsets,
    cudf::column_view const& poly2_xs,
    cudf::column_view const& poly2_ys,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    // TODO
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }
};

}  // namespace

namespace cuspatial {

std::unique_ptr<cudf::column> pairwise_polygon_distance(
  cudf::device_span<cudf::size_type> const& poly1_offsets,
  cudf::device_span<cudf::size_type> const& poly1_ring_offsets,
  cudf::column_view const& poly1_xs,
  cudf::column_view const& poly1_ys,
  cudf::device_span<cudf::size_type> const& poly2_offsets,
  cudf::device_span<cudf::size_type> const& poly2_ring_offsets,
  cudf::column_view const& poly2_xs,
  cudf::column_view const& poly2_ys,
  rmm::mr::device_memory_resource* mr)
{
  using device_span_size_type = cudf::device_span<cudf::size_type>::size_type;

  CUSPATIAL_EXPECTS(poly1_xs.type() == poly1_ys.type(),
                    "Inputs `poly1_xs` and `poly1_ys` must have same type.");
  CUSPATIAL_EXPECTS(poly2_xs.type() == poly2_ys.type(),
                    "Inputs `poly2_xs` and `poly2_ys` must have same type.");
  CUSPATIAL_EXPECTS(poly1_xs.type() == poly2_ys.type(),
                    "Polygons of the same pair must have the same type.");
  CUSPATIAL_EXPECTS(poly1_xs.size() == poly1_ys.size(),
                    "Inputs `xs` and `ys` must have same length.");
  CUSPATIAL_EXPECTS(poly2_xs.size() == poly2_ys.size(),
                    "Inputs `xs` and `ys` must have same length.");

  CUSPATIAL_EXPECTS(not poly1_xs.has_nulls() and not poly1_ys.has_nulls() and
                      not poly2_xs.has_nulls() and not poly2_ys.has_nulls(),
                    "Coordinates must not have nulls.");

  CUSPATIAL_EXPECTS(
    static_cast<device_span_size_type>(poly1_xs.size()) >= poly1_ring_offsets.size(),
    "At least 1 point is required for each linear ring of the polygon.");
  CUSPATIAL_EXPECTS(
    static_cast<device_span_size_type>(poly2_xs.size()) >= poly2_ring_offsets.size(),
    "At least 1 point is required for each linear ring of the polygon.");

  return cudf::type_dispatcher(poly1_xs.type(),
                               polygon_distance_functor{},
                               poly1_offsets,
                               poly1_ring_offsets,
                               poly1_xs,
                               poly1_ys,
                               poly2_offsets,
                               poly2_ring_offsets,
                               poly2_xs,
                               poly2_ys,
                               rmm::cuda_stream_default,
                               mr);
}

}  // namespace cuspatial
