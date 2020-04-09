/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>

namespace cuspatial {

/**
 * @brief Translates lon/lat relative to origin and converts to x/y coordinates.
 *
 * @param[in] origin_lon: longitude of origin
 * @param[in] origin_lat: latitude of origin
 * @param[in] input_lon: longitudes to transform
 * @param[in] input_lat: latitudes to transform
 *
 * @returns a pair of columns storing transformed x/y coordinates
 */

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
lonlat_to_coord(double origin_lon,
                double origin_lat,
                cudf::column_view const& input_lon,
                cudf::column_view const& input_lat,
                rmm::mr::device_memory_resource *mr =
                  rmm::mr::get_default_resource());

}  // namespace cuspatial
