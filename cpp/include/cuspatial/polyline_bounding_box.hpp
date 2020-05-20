/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
 * @brief compute bounding boxes (bboxes) of a set of polygons
 *
 * @param spos polyline offset array to vertices
 * @param x polygon x coordiante array.
 * @param y polygon y coordiante array.
 * @param R expansion radius
 *
 * @return experimental::table with four arrays of bounding boxes, x1,y1,x2,y2.
 */

std::unique_ptr<cudf::experimental::table> polyline_bbox(
  cudf::column_view const& spos,
  cudf::column_view const& x,
  cudf::column_view const& y,
  double R,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
