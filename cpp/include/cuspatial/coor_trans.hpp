#pragma once
#include <cudf/cudf.h>

namespace cuSpatial {


/**
 * @Brief transforming in_lon/in_lat (lon/lat defined in Coord) to out_x/out_y relative to a camera origiin

 * @param[in] cam_lon/cam_lat: x and y coordiante of camera origin (lon/lat)

 * @param[in] in_lon/in_lat: arrays of x and y coordiantes (lon/lat) of input points to be transformed

 * @param[out] out_x/out_y: arrays of x and y coordiantes after transformation (in kilometers -km)
 */
void ll2coor(const gdf_scalar  & cam_lon,const gdf_scalar  & cam_lat,const gdf_column  & in_lon,const gdf_column  & in_lat,
    	gdf_column & out_x,gdf_column & out_y /* ,cudaStream_t stream = 0   */);

}  // namespace cuSpatial

