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

typedef struct gdf_column_ gdf_column;  // forward declaration

namespace cuspatial {

/**
 * brief Compute Haversine distances among pairs of longitude/latitude locations
 *
 * @param[in] x1: longitude coordinates of the starting points
 * @param[in] y1: latitude coordinates of the starting points
 * @param[in] x2: longitude coordinates of the ending points
 * @param[in] y2: latitude coordinates of the ending points
 *
 * @return array of distances in kilometers (km) for all (x1,y1) and (x2,y2)
 *         point pairs
 */
gdf_column haversine_distance(const gdf_column& x1,
                              const gdf_column& y1,
                              const gdf_column& x2,
                              const gdf_column& y2);

}  // namespace cuspatial
