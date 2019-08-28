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
#include <cuspatial/cuspatial.h>

namespace cuspatial {

	/**
	*@Brief read uint32_t data from file as column; mostly for identifiers, lengths and positions.
    *mostly for identifiers, lengths and positions.
    * @param[in] filename: file to read
    * @return gdf_column storing the uint32_t data
	*/
	void read_uint_soa(const char *col_fn, gdf_column& ids);


	/**
	 * @Brief read timestamp (ts: its_timestamp type) data from file as column
     *@param[in] ts_fn: file name of timestamp column data
     *@param[out] ids: gdf_column storing the timestamp column
	*/
	void read_ts_soa(const char *ts_fn, gdf_column& ts);

	/**
	 * @Brief read lon/lat from file as two columns; data type is fixed to double (GDF_FLOAT64)
     *@param[in] pnt_fn: file name of point data in location_3d layout (lon/lat/alt but alt is omitted)
     *@param[out] pnt_lon: gdf_column storing the longitude column
     *@param[out] pnt_lat: gdf_column storing the latitude column
	*/
	void read_pnt_lonlat_soa(const char *pnt_fn, gdf_column& pnt_lon,gdf_column& pnt_lat);


	/**
	 *@Brief read x/y from file as two columns; data type is fixed to double (GDF_FLOAT64)
     *@param[in] pnt_fn: file name of point data in coord_2d layout (x/y)
     *@param[out] pnt_x: gdf_column storing the x column
     *@param[out] pnt_y: gdf_column storing the y column
     Note: x/y can be lon/lat.
	*/
	void read_pnt_xy_soa(const char *pnt_fn, gdf_column& pnt_x,gdf_column& pnt_y);


	/**
	 * @Brief read poygon data from file in SoA format; data type of vertices is fixed to double (GDF_FLOAT64)
	 * @param[in] ply_fn: polygon data file name
	 * @param[out] ply_fpos: pointer/array to index polygons, i.e., prefix-sum of #of rings of all polygons
	 * @param[out] ply_rpos: pointer/array to index rings, i.e., prefix-sum of #of vertices of all rings
	 * @param[out] ply_x: pointer/array of x coordinates of concatenated polygons
	 * @param[out] ply_y: pointer/array of x coordinates of concatenated polygons
	 Note: x/y can be lon/lat.
	*/
	void read_ply_soa(const char *ply_fn,gdf_column& ply_fpos, gdf_column& ply_rpos,
									   gdf_column& ply_x,gdf_column& ply_y);


}// namespace cuspatial
