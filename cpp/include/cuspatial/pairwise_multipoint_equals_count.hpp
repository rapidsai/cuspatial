/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial/column/geometry_column_view.hpp>

#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup spatial
 * @brief Count the number of equal points in pairs of multipoints..
 *
 * Given two columns of multipoints, returns a column containing the
 * count of points in each multipoint from `lhs` that exist in the
 * corresponding multipoint in `rhs`.
 *
 * @param lhs Geometry column of multipoints with interleaved coordinates
 * @param rhs Geometry column of multipoints with interleaved coordinates
 * @param mr Device memory resource used to allocate the returned column.
 * @return A column of size len(lhs) containing the number of points in each
 * multipoint from `lhs` that are equal to a point in the corresponding
 * multipoint in `rhs`.
 *
 * @throw cuspatial::logic_error if `lhs` and `rhs` have different coordinate
 * types or lengths.
 *
 * @example
 * ```
 * lhs: MultiPoint(0, 0)
 * rhs: MultiPoint((0, 0), (1, 1), (2, 2), (3, 3))
 * result: 1

 * lhs: MultiPoint((0, 0), (1, 1), (2, 2), (3, 3))
 * rhs: MultiPoint((0, 0))
 * result: 1

 * lhs: (
 *        MultiPoint((3, 3), (3, 3), (0, 0)),
 *        MultiPoint((0, 0), (1, 1), (2, 2)),
 *        MultiPoint((0, 0))
 *      )
 * rhs: (
 *        MultiPoint((0, 0), (2, 2), (1, 1)),
 *        MultiPoint((2, 2), (0, 0), (1, 1)),
 *        MultiPoint((1, 1))
 *      )
 * result: ( 1, 3, 0 )
 */
std::unique_ptr<cudf::column> pairwise_multipoint_equals_count(
  geometry_column_view const& lhs,
  geometry_column_view const& rhs,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
