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

#include <memory>
#include <cudf/column/column.hpp>
#include <cudf/column_view/column_view.hpp>
#include "cudf/types.hpp"

namespace cuspatial {

/**
 * @brief Point-in-Polygon (PIP) tests between a column of points and a
 *        column of polygons
 *
 * Inputs follow the ESRI shapefile format. Polygons are represented by a set of rings, which are in turn represented
 * by a set of line segments. Clockwise rings represent a positive space, and anti-clockwise rings represent a
 * negative space.
 *
 * @param[in] test_points_x:     x component of target points
 * @param[in] test_points_y:     y component of target points
 * @param[in] poly_offsets:      beginning index of the first ring in each polygon
 * @param[in] poly_ring_offsets: beginning index of the first point in each ring
 * @param[in] poly_points_x:     x component of polygon points
 * @param[in] poly_points_y:     y component of polygon points
 *
 * @returns gdf_column of type GDF_INT32; the jth bit of the ith element of the
 *          returned int32_t column is set if the ith point is in the jth polygon
 *
 * Note: The Number of polygons, i.e. (poly_offsets.size) cannot exceed mask size of
 * `32 == sizeof(uint32_t) * 8`.
 */
std::unique_ptr<cudf::column>
point_in_polygon_bitmap(cudf::column_view const& test_points_x,
                        cudf::column_view const& test_points_y,
                        cudf::column_view const& poly_offsets,
                        cudf::column_view const& poly_ring_offsets, 
                        cudf::column_view const& poly_points_x,
                        cudf::column_view const& poly_points_y,
                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
