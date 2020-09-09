/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <memory>

namespace cuspatial {

/**
 * @brief Compute minimum bounding boxes of a set of polylines and an expansion radius.
 *
 * @param poly_offsets Begin indices of the first point in each polyline (i.e. prefix-sum)
 * @param x Polyline point x-coordinates
 * @param y Polyline point y-coordinates
 * @param expansion_radius Radius of each polyline point
 *
 * @return a cudf table of bounding boxes as four columns of the same type as `x` and `y`:
 * x_min - the minimum x-coordinate of each bounding box
 * y_min - the minimum y-coordinate of each bounding box
 * x_max - the maximum x-coordinate of each bounding box
 * y_max - the maximum y-coordinate of each bounding box
 */

std::unique_ptr<cudf::table> polyline_bounding_boxes(
  cudf::column_view const& poly_offsets,
  cudf::column_view const& x,
  cudf::column_view const& y,
  double expansion_radius,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
