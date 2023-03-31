/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup spatial
 * @brief Compute the number of pairs of multipoints that are equal.
 *
 * Given two columns of interleaved multipoint coordinates, returns a column
 * containing the count of points in each multipoint from `lhs` that are equal
 * to a point in the corresponding multipoint in `rhs`.
 *
 * @param lhs Geometry column with a multipoint of interleaved coordinates
 * @param rhs Geometry column with a multipoint of interleaved coordinates
 * @param mr Device memory resource used to allocate the returned column.
 * @return A column of size len(lhs) containing the count that each point of
 * the multipoint in `lhs` is equal to a point in `rhs`.
 *
 * @throw cuspatial::logic_error if `lhs` and `rhs` have different coordinate
 * types.
 *
 * @example
 * ```
 * lhs: 0, 0, 1, 1, 2, 2
 * rhs: 0, 0, 1, 1, 2, 2
 * result: 1, 1, 1
 *
 * lhs: 0, 0, 1, 1, 2, 2
 * rhs: 0, 0
 * result: 1, 0, 0
 */

/**
 */
std::unique_ptr<cudf::column> allpairs_multipoint_equals_count(
  cudf::column_view const& lhs,
  cudf::column_view const& rhs,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
