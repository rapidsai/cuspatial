#pragma once

#include <cudf/cudf.h>
#include <cuspatial/cuspatial.h>

namespace cuSpatial {

	/**
	*@Brief read uint32_t (unsigned integer with 32 bit fixed length) data from file as column;
    *mostly for identifiers, lengths and positions.
    *@param[in] col_fn: file name of uint32_t column data
    *@param[out] ids: gdf_column storing the uint32_t column
	*/
	void read_uint_soa(const char *col_fn, gdf_column& ids);


	/**
	 * @Brief read timestamp (ts: TimeStamp type) data from file as column
     *@param[in] ts_fn: file name of timestamp column data
     *@param[out] ids: gdf_column storing the timestamp column
	*/
	void read_ts_soa(const char *ts_fn, gdf_column& ts);

	/**
	 * @Brief read lon/lat from file as two columns; data type is fixed to double (GDF_FLOAT64)
     *@param[in] pnt_fn: file name of point data in Location3D layout (lon/lat/alt but alt is omitted)
     *@param[out] pnt_lon: gdf_column storing the longitude column
     *@param[out] pnt_lat: gdf_column storing the latitude column
	*/
	void read_pnt_lonlat_soa(const char *pnt_fn, gdf_column& pnt_lon,gdf_column& pnt_lat);


	/**
	 *@Brief read x/y from file as two columns; data type is fixed to double (GDF_FLOAT64)
     *@param[in] pnt_fn: file name of point data in Coord2D layout (x/y)
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
	 * @param[out] ply_x: pointer/array of x coordiantes of concatenated polygons
	 * @param[out] ply_y: pointer/array of x coordiantes of concatenated polygons
	 Note: x/y can be lon/lat.
	*/
	void read_ply_soa(const char *ply_fn,gdf_column& ply_fpos, gdf_column& ply_rpos,
									   gdf_column& ply_x,gdf_column& ply_y);


}// namespace cuSpatial
