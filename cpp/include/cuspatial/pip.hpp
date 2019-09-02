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
 * @brief Point-in-Polygon (PIP) tests between a column of points and a
 *        column of polygons
 *
 * @param[in] pnt_x: x coordinates of points
 * @param[in] pnt_y: y coordinates of points
 * @param[in] ply_fpos: index polygons: prefix sum of number of rings of all
 *            polygons
 * @param[in] ply_rpos: index rings: prefix sum of  number of vertices of all
 *            rings
 * @param[in] ply_x: x coordinates of concatenated polygons
 * @param[in] ply_y: y coordinates of concatenated polygons
 *
 * @returns gdf_column of type GDF_INT32; the jth bit of the ith element of the
 *          returned GDF_INT32 array is 1 if the ith point is in the jth polygon
 *
 * Note: The # of polygons, i.e., ply_fpos.size cannot exceed 
 * 32 == sizeof(uint32_t)*8. It is possible to use larger integers to
 * accommodate more polygons (e.g., 64/128) in the future. For more polygons,
 * the polygons need to be indexed and the problem essentially becomes a spatial
 * join.
 */
gdf_column pip_bm(const gdf_column& pnt_x, const gdf_column& pnt_y,
                  const gdf_column& ply_fpos, const gdf_column& ply_rpos,
                  const gdf_column& ply_x, const gdf_column& ply_y);

}  // namespace cuspatial
