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
 * @brief Tests whether the specified points are inside any of the specified polygons.
 *
 * Tests whether points are inside at most 31 polygons. Polygons are a collection of one or more
 * rings. Rings are a collection of three or more vertices.
 *
 * @param[in] test_points_x:     x-coordinates of points to test
 * @param[in] test_points_y:     y-coordinates of points to test
 * @param[in] poly_offsets:      beginning index of the first ring in each polygon
 * @param[in] poly_ring_offsets: beginning index of the first point in each ring
 * @param[in] poly_points_x:     x-coordinates of polygon points
 * @param[in] poly_points_y:     y-coordinates of polygon points
 *
 * @returns A column of INT32 containing one element per input point. Each bit (except the sign bit)
 * represents a hit or miss for each of the input polygons in least-significant-bit order. i.e.
 * `output[3] & 0b0010` indicates a hit or miss for the 3rd point against the 2nd polygon.
 *
 * @note Limit 31 polygons per call. Polygons may contain multiple rings.
 * @note Direction of rings does not matter.
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
std::pair<std::unique_ptr<cudf::column>, cudf::table_view> byte_point_in_polygon(
  cudf::column_view const& test_points_x,
  cudf::column_view const& test_points_y,
  cudf::column_view const& poly_offsets,
  cudf::column_view const& poly_ring_offsets,
  cudf::column_view const& poly_points_x,
  cudf::column_view const& poly_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Tests whether the specified points are inside any of the specified polygons.
 *
 * Tests whether points are inside an unlimited number of polygons. Instead of being
 * limited by an output byte mask, is limited by the multiple of the polygons size and
 * point size, which must remain beneat UINT32::Max. Polygons are a collection of one or more
 * rings. Rings are a collection of three or more vertices.
 *
 * @param[in] test_points_x:     x-coordinates of points to test
 * @param[in] test_points_y:     y-coordinates of points to test
 * @param[in] poly_offsets:      beginning index of the first ring in each polygon
 * @param[in] poly_ring_offsets: beginning index of the first point in each ring
 * @param[in] poly_points_x:     x-coordinates of polygon points
 * @param[in] poly_points_y:     y-coordinates of polygon points
 *
 * @returns a std::pair of an owning column and a non-owning table_view. The table_view contains
 * one column for each polygon. Each row is the result of testing the corresponding point in
 * `test_points_x` and `test_points_y` against the corresponding polygon.
 * represents a hit or miss for each of the input polygons in least-significant-bit order. i.e.
 * `output[3] & 0b0010` indicates a hit or miss for the 3rd point against the 2nd polygon.
 *
 * @note Is slower and handles fewer points than the `byte_point_in_polygon` function.
 * @note Direction of rings does not matter.
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
std::pair<std::unique_ptr<cudf::column>, cudf::table_view> columnar_point_in_polygon(
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
