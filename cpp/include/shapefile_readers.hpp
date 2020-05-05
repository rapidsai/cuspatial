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

#include <cudf/table/table.hpp>
#include <cudf/column/column_view.hpp>

namespace cuspatial {

/**
 * @brief read polygon data from an ESRI Shapefile.
 *
 * data type of vertices is fixed to double (GDF_FLOAT64)
 *
 * @param[in] filename: polygon data filename in ESRI Shapefile format
 * @param[out] poly_offsets: index polygons: prefix sum of number of rings of all
 *             polygons
 * @param[out] poly_ring_offsets: index rings: prefix sum of number of vertices of all
 *             rings
 * @param[out] poly_points_x: x coordinates of concatenated polygons
 * @param[out] poly_points_y: y coordinates of concatenated polygons
 *
 * @note: x/y can be lon/lat.
**/
std::unique_ptr<cudf::table> read_polygon_shapefile(const char *filename);

} // namespace cuspatial
