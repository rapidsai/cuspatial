/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup spatial_relationship
 * @{
 */

/**
 * @brief Compute minimum bounding boxes of a set of linestrings and an expansion radius.
 *
 * @param linestring_offsets Begin indices of the first point in each linestring (i.e. prefix-sum)
 * @param x Linestring point x-coordinates
 * @param y Linestring point y-coordinates
 * @param expansion_radius Radius of each linestring point
 *
 * @return a cudf table of bounding boxes as four columns of the same type as `x` and `y`:
 * x_min - the minimum x-coordinate of each bounding box
 * y_min - the minimum y-coordinate of each bounding box
 * x_max - the maximum x-coordinate of each bounding box
 * y_max - the maximum y-coordinate of each bounding box
 *
 * @pre For compatibility with GeoArrow, the size of @p linestring_offsets should be one more than
 * the number of linestrings to process. The final offset is not used by this function, but the
 * number of offsets determines the output size.
 */

std::unique_ptr<cudf::table> linestring_bounding_boxes(
  cudf::column_view const& linestring_offsets,
  cudf::column_view const& x,
  cudf::column_view const& y,
  double expansion_radius,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute minimum bounding box for each polygon in a list.
 *
 * @param poly_offsets Begin indices of the first ring in each polygon (i.e. prefix-sum)
 * @param ring_offsets Begin indices of the first point in each ring (i.e. prefix-sum)
 * @param x Polygon point x-coordinates
 * @param y Polygon point y-coordinates
 * @param expansion_radius radius to add to each point when computing its bounding box.
 *
 * @return a cudf table of bounding boxes as four columns of the same type as `x` and `y`:
 * x_min - the minimum x-coordinate of each bounding box
 * y_min - the minimum y-coordinate of each bounding box
 * x_max - the maximum x-coordinate of each bounding box
 * y_max - the maximum y-coordinate of each bounding box
 *
 * @pre For compatibility with GeoArrow, the size of @p poly_offsets should be one more than the
 * number of polygons to process. The size of @p ring_offsets should be one more than the number of
 * total rings. The final offset in each range is not used by this function, but the number of
 * polygon offsets determines the output size.
 */
std::unique_ptr<cudf::table> polygon_bounding_boxes(
  cudf::column_view const& poly_offsets,
  cudf::column_view const& ring_offsets,
  cudf::column_view const& x,
  cudf::column_view const& y,
  double expansion_radius           = 0.0,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
