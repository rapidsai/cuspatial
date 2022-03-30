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

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

namespace cuspatial {

/**
 * @brief Compute the shortest distance between pairs of polygons.
 *
 * The shortest distance between two polygons is computed with the shortest distance
 * between every pair of vertex, line segments in two polygons. If two polygons intersects,
 * or one polygon contains another, the distance is 0.
 *
 * @param [in] poly1_offsets The indices that indicates the start coordinate of the polygon
 * @param [in] poly1_ring_offsets The indices that indicates the start coordinate of the polygon
 * @param [in] poly1_xs The x-coordinates of input points
 * @param [in] poly1_ys The y-coordinates of input points
 * @param [in] poly2_offsets The indices that indicates the start coordinate of the polygon
 * @param [in] poly2_ring_offsets The indices that indicates the start coordinate of the polygon
 * @param [in] poly2_xs The x-coordinates of input points
 * @param [in] poly2_ys The y-coordinates of input points
 * @param [in] mr Memory resource allocating memory to store the result column
 *
 * @throws cuspatial::logic_error if the poly1_offsets != poly2_offsets, unmatching number of
 * polygons in polygon list 1 and polygon list 2.
 * @throws cuspatial::logic_error if any of poly1_xs, poly1_ys, poly2_xs, poly2_ys have different
 * types.
 * @throws cuspatial::logic_error if any of poly1_xs, poly1_ys, poly2_xs, poly2_ys contains null.
 *
 * @return A cudf::column that contains the shortest distances
 */

std::unique_ptr<cudf::column> pairwise_polygon_distance(
  cudf::device_span<cudf::size_type> const& poly1_offsets,
  cudf::device_span<cudf::size_type> const& poly1_ring_offsets,
  cudf::column_view const& poly1_xs,
  cudf::column_view const& poly1_ys,
  cudf::device_span<cudf::size_type> const& poly2_offsets,
  cudf::device_span<cudf::size_type> const& poly2_ring_offsets,
  cudf::column_view const& poly2_xs,
  cudf::column_view const& poly2_ys,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
