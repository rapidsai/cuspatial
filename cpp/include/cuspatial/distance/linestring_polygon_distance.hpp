/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>

#include <optional>

namespace cuspatial {

/**
 * @ingroup distance
 * @brief Compute pairwise (multi)linestring-to-(multi)polygon Cartesian distance
 *
 * @param multilinestrings Geometry column of multilinestrings
 * @param multipolygons Geometry column of multipolygons
 * @param mr Device memory resource used to allocate the returned column.
 * @return Column of distances between each pair of input geometries, same type as input coordinate
 * types.
 *
 * @throw cuspatial::logic_error if `multilinestrings` and `multipolygons` has different coordinate
 * types.
 * @throw cuspatial::logic_error if `multilinestrings` is not a linestring column and `multipolygons` is not a
 * polygon column.
 * @throw cuspatial::logic_error if input column sizes mismatch.
 */

std::unique_ptr<cudf::column> pairwise_linestring_polygon_distance(
  geometry_column_view const& multilinestrings,
  geometry_column_view const& multipolygons,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
