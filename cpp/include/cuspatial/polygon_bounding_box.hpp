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
 * @brief Compute bounding boxes for a set of polygons
 *
 * @param poly_offsets Polygon to first ring offset
 * @param ring_offsets Ring to first point offset
 * @param x Polygon x-coordinates
 * @param y Polygon y-coordinates
 *
 * @return a cudf table of bounding boxes as four columns of the same type as `x` and `y`:
 * x1 - the lower-left x-coordinate of each bounding box
 * y1 - the lower-left y-coordinate of each bounding box
 * x2 - the upper-right x-coordinate of each bounding box
 * y2 - the upper-right y-coordinate of each bounding box
 */

std::unique_ptr<cudf::table> polygon_bounding_boxes(
  cudf::column_view const& poly_offsets,
  cudf::column_view const& ring_offsets,
  cudf::column_view const& x,
  cudf::column_view const& y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
