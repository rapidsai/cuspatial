/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <memory>

namespace cuspatial {

/**
 * @brief Compute minimum bounding boxes of a set of linestrings and an expansion radius.
 *
 * @ingroup spatial_relationship
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
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
