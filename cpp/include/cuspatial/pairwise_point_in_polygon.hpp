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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup spatial_relationship
 * @{
 */

/**
 * @brief Given (point, polygon pairs), tests whether the point of each pair is inside the polygon
 * of the pair.
 *
 * Tests that each point is or is not inside of the polygon in the corresponding index.
 * Polygons are a collection of one or more * rings. Rings are a collection of three or more
 * vertices.
 *
 * @param[in] test_points_x:     x-coordinates of points to test
 * @param[in] test_points_y:     y-coordinates of points to test
 * @param[in] poly_offsets:      beginning index of the first ring in each polygon
 * @param[in] poly_ring_offsets: beginning index of the first point in each ring
 * @param[in] poly_points_x:     x-coordinates of polygon points
 * @param[in] poly_points_y:     y-coordinates of polygon points
 *
 * @returns A column of booleans for each point/polygon pair.
 *
 * @note Direction of rings does not matter.
 * @note Supports open or closed polygon formats.
 * @note This algorithm supports the ESRI shapefile format, but assumes all polygons are "clean" (as
 * defined by the format), and does _not_ verify whether the input adheres to the shapefile format.
 * @note Overlapping rings negate each other. This behavior is not limited to a single negation,
 * allowing for "islands" within the same polygon.
 * @note `poly_ring_offsets` must contain only the rings that make up the polygons indexed by
 * `poly_offsets`. If there are rings in `poly_ring_offsets` that are not part of the polygons in
 * `poly_offsets`, results are likely to be incorrect and behavior is undefined.
 *
 * ```
 *   poly w/two rings         poly w/four rings
 * +-----------+          +------------------------+
 * :███████████:          :████████████████████████:
 * :███████████:          :██+------------------+██:
 * :██████+----:------+   :██:  +----+  +----+  :██:
 * :██████:    :██████:   :██:  :████:  :████:  :██:
 * +------;----+██████:   :██:  :----:  :----:  :██:
 *        :███████████:   :██+------------------+██:
 *        :███████████:   :████████████████████████:
 *        +-----------+   +------------------------+
 * ```
 */
std::unique_ptr<cudf::column> pairwise_point_in_polygon(
  cudf::column_view const& test_points_x,
  cudf::column_view const& test_points_y,
  cudf::column_view const& poly_offsets,
  cudf::column_view const& poly_ring_offsets,
  cudf::column_view const& poly_points_x,
  cudf::column_view const& poly_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
