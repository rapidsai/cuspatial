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
 * @addtogroup projection
 * @{
 */

/**
 * @brief Sinusoidal projection of longitude/latitude relative to origin to Cartesian (x/y)
 * coordinates in km.
 *
 * Can be used to approximately convert longitude/latitude coordinates to Cartesian coordinates
 * given that all points are near the origin. Error increases with distance from the origin.
 * See [Sinusoidal Projection](https://en.wikipedia.org/wiki/Sinusoidal_projection) for more detail.
 *
 * @param origin_lon: longitude of origin
 * @param origin_lat: latitude of origin
 * @param input_lon: longitudes to transform
 * @param input_lat: latitudes to transform
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @returns a pair of columns containing cartesian coordinates in kilometers
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> sinusoidal_projection(
  double origin_lon,
  double origin_lat,
  cudf::column_view const& input_lon,
  cudf::column_view const& input_lat,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
