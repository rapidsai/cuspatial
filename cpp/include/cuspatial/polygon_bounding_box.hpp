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
 * @brief Compute minimum bounding boxes for a set of polygons.
 *
 * @param poly_offsets Begin indices of the first ring in each polygon (i.e. prefix-sum)
 * @param ring_offsets Begin indices of the first point in each ring (i.e. prefix-sum)
 * @param x Polygon point x-coordinates
 * @param y Polygon point y-coordinates
 *
 * @return a cudf table of bounding boxes as four columns of the same type as `x` and `y`:
 * x_min - the minimum x-coordinate of each bounding box
 * y_min - the minimum y-coordinate of each bounding box
 * x_max - the maximum x-coordinate of each bounding box
 * y_max - the maximum y-coordinate of each bounding box
 */

std::unique_ptr<cudf::table> polygon_bounding_boxes(
  cudf::column_view const& poly_offsets,
  cudf::column_view const& ring_offsets,
  cudf::column_view const& x,
  cudf::column_view const& y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
