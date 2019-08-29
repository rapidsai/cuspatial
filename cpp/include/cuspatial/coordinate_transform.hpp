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
#include <cudf/cudf.h>

namespace cuspatial {


/**
 * @brief transform 2D longitude/latitude coordinates to x/y coordinates relative to a camera origin

 * @param[in] cam_lon: longitude of camera origin
 * @param[in] cam_lat: latitude of camera origin
 * @param[in] in_lon: column of longitude coordinates of input points to be transformed
 * @param[in] in_lat: column of latitude coordinates of input points to be transformed
 * @param[out] out_x: column of x coordinates after transformation in kilometers (km)
 * @param[out] out_y: column of y coordinates after transformation in kilometers (km)
 */
std::pair<gdf_column,gdf_column> lonlat_to_coord(const gdf_scalar& cam_lon, const gdf_scalar& cam_lat,
	const gdf_column& in_lon, const gdf_column  & in_lat);

}  // namespace cuspatial
