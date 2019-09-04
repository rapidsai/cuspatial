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

#include <cudf/types.h>

namespace cuspatial {

/**
 * @brief Find all points (x,y) that fall within a query window
 * (left, bottom, right, top)

 * @param[in] left:   x-coordinate of left edge of the query window
 * @param[in] bottom: y-coordinate of bottom of the query window
 * @param[in] right:  x-coordinate of right edge of the query window
 * @param[in] top:    y-coordinate of top of the query window
 * @param[in] x:      x-coordinates of points to be queried
 * @param[in] y:      y-coordinates of points to be queried

 * @returns pair of gdf_columns of query results of in_x and in_y columns.
 */

 std::pair<gdf_column,gdf_column> spatial_window_points(const gdf_scalar& x1,
                                                        const gdf_scalar& y1,
                                                        const gdf_scalar& x2,
                                                        const gdf_scalar& y2,
                                                        const gdf_column& in_x,
                                                        const gdf_column& in_y);

}  // namespace cuspatial
