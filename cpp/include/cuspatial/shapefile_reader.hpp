/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace cuspatial {

typedef enum class winding_order : bool { CLOCKWISE, COUNTER_CLOCKWISE } winding_order;

/**
 * @addtogroup io
 * @{
 */

/**
 * @brief read polygon data from an ESRI Shapefile.
 *
 * @param[in] filename: ESRI Shapefile file path (usually ends in .shp)
 * @param[in] outer_ring_winding: the ordering of the outer ring of polygon vertices; clockwise or
 * counter-clockwise
 * @param[in] mr:       Optional, The resource to use to allocate the returned data
 *
 * @return Vector of 4 columns representing one or more polygons:
 *  - Column 0, INT32:   beginning index of the first ring in each polygon
 *  - Column 1, INT32:   beginning index of the first point in each ring
 *  - Column 2, FLOAT64: x component of polygon points
 *  - Column 3, FLOAT64: y component of polygon points
 *
 * @note The number of polygons is equal to the length of the first column
 *
 **/
[[deprecated(
  "Use Python libraries or GDAL to load shapefiles.")]] std::vector<std::unique_ptr<cudf::column>>
read_polygon_shapefile(
  std::string const& filename,
  const winding_order outer_ring_winding = winding_order::COUNTER_CLOCKWISE,
  rmm::mr::device_memory_resource* mr    = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
