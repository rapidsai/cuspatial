#pragma once

#include <cudf/cudf.h>

namespace cuSpatial {

/**
 * @Brief retrive all points (x,y) that fall within a query window (x1,y1,x2,y2) and output the filtered points

 * @param[in] x1/x2/y1/y2: defines the query window

 * @param[in] in_x/in_y: pointer/array of x/y coordiantes of points to be queried

 * @param[out] out_x/out_y: pointer/array of x/y coordiantes of points that fall within the query window

 * @returns the number of points that satisfy the spatial window query criteria
 */

int sw_xy(const gdf_scalar x1,const gdf_scalar y1,const gdf_scalar x2,const gdf_scalar y2,
	const gdf_column& in_x,const gdf_column& in_y, gdf_column& out_x,gdf_column& out_y);

}  // namespace cuSpatial

