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

typedef struct gdf_column_ gdf_column; // forward declaration

namespace cuspatial {

/**
 * @brief read polygon data from file in SoA format
 *
 * data type of vertices is fixed to double (GDF_FLOAT64)
 *
 * @param[in] filename: polygon data filename in ESRI Shapefile format
 * @param[out] ply_fpos: index polygons: prefix sum of number of rings of all
 *             polygons
 * @param[out] ply_rpos: index rings: prefix sum of number of vertices of all
 *             rings
 * @param[out] ply_x: x coordinates of concatenated polygons
 * @param[out] ply_y: y coordinates of concatenated polygons
 *
 * @note: x/y can be lon/lat.
**/
void read_polygon_shp(const char *filename,
                      gdf_column* ply_fpos, gdf_column* ply_rpos,
                      gdf_column* ply_x, gdf_column* ply_y);

}// namespace cuspatial
