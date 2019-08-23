#pragma once
#include <cudf/cudf.h>

namespace cuspatial {


/**
 * @Brief transform 2D longitude/latitude coordinates to x/y coordinates relative to a camera origin

 * @param[in] cam_lon: longitude of camera origin
 * @param[in] cam_lat: latitude of camera origin
 * @param[in] in_lon: column of longitude coordinates of input points to be transformed
 * @param[in] in_lat: column of latitude coordinates of input points to be transformed
 * @param[out] out_x: column of x coordinates after transformation in kilometers (km)
 * @param[out] out_y: column of y coordinates after transformation in kilometers (km)
 */
void lonlat_to_coord(const gdf_scalar& cam_lon, const gdf_scalar& cam_lat, const gdf_column& in_lon, const gdf_column  & in_lat,
    	gdf_column & out_x, gdf_column & out_y);

}  // namespace cuspatial

