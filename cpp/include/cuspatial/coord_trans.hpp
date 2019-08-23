#pragma once
#include <cudf/cudf.h>

namespace cuSpatial {


/**
 * @Brief transform 2D longitude/latitude coordinates to x/y coordinates relative to a camera origin

 * @param[in] cam_lon: longitude of camera origin
 * @param[in] cam_lat: latitude of camera origin
 * @param[in] in_lon: array of longitude coordinates of input points to be transformed
 * @param[in] in_lat: array of latitude coordinates of input points to be transformed
 * @param[out] out_x: array of x coordiantes after transformation (in kilometers -km)
 * @param[out] out_y: array of y coordiantes after transformation (in kilometers -km)
 */
void lonlat_to_coord(const gdf_scalar  & cam_lon,const gdf_scalar  & cam_lat,const gdf_column  & in_lon,const gdf_column  & in_lat,
    	gdf_column & out_x,gdf_column & out_y /* ,cudaStream_t stream = 0   */);

}  // namespace cuSpatial

