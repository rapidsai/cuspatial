#pragma once
#include <cudf/cudf.h>

namespace cuSpatial {

/**
 * @Brief Compute Haversine distances among pairs of logitude/latitude locations

 * @param[in] x1: pointer/array of longitude coordiantes of the starting points
 * @param[in] y1: pointer/array of latitude  coordiantes of the starting points
 * @param[in] x2: pointer/array of longitude coordiantes of the ending points
 * @param[in] y2: pointer/array of latitude coordiantes of the ending points

 * @returns the an array of distances (in kilometers -km) for all (x1,y1) and (x2,y2) point pairs

 */

gdf_column haversine_distance(const gdf_column& x1,const gdf_column& y1,const gdf_column& x2,const gdf_column& y2
                          /* , cudaStream_t stream = 0   */);

}  // namespace cuSpatial

