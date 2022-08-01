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

#include <cuspatial/constants.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace cuspatial {

/**
 * @brief Compute haversine distances between points in set A and the corresponding points in set B.
 *
 * @ingroup distance
 *
 * https://en.wikipedia.org/wiki/Haversine_formula
 *
 * @param[in]  a_lon: longitude of points in set A
 * @param[in]  a_lat:  latitude of points in set A
 * @param[in]  b_lon: longitude of points in set B
 * @param[in]  b_lat:  latitude of points in set B
 * @param[in] radius: radius of the sphere on which the points reside. default: 6371.0 (aprx. radius
 * of earth in km)
 *
 * @return array of distances for all (a_lon[i], a_lat[i]) and (b_lon[i], b_lat[i]) point pairs
 */
std::unique_ptr<cudf::column> haversine_distance(
  cudf::column_view const& a_lon,
  cudf::column_view const& a_lat,
  cudf::column_view const& b_lon,
  cudf::column_view const& b_lat,
  double const radius                 = EARTH_RADIUS_KM,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
