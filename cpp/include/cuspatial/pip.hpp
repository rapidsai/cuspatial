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
 * @brief Point-in-Polygon (PIP) tests among a vector/array of points and a vector/array of polygons

 * @param[in] pnt_x: pointer/array of x coordinates of points
 * @param[in] pnt_y: pointer/array of y coordinates of points
 * @param[in] ply_fpos: pointer/array to index polygons, i.e., prefix-sum of #of rings of all polygons
 * @param[in] ply_rpos: pointer/array to index rings, i.e., prefix-sum of #of vertices of all rings
 * @param[in] ply_x: pointer/array of x coordinates of concatenated polygons
 * @param[in] ply_y: pointer/array of x coordinates of concatenated polygons
 *
 * @returns gdf_column of type GDF_INT32; the jth bit of the ith element of the returned GDF_INT32 array indicate
 * whether the ith point is in the jth polygon.

 * Note: The # of polygons, i.e., ply_fpos.size can not exceed sizeof(uint)*8, i.e., 32.
 */

gdf_column pip_bm(const gdf_column& pnt_x,const gdf_column& pnt_y,
                                   const gdf_column& ply_fpos, const gdf_column& ply_rpos,
                                   const gdf_column& ply_x,const gdf_column& ply_y);


}  // namespace cuspatial
