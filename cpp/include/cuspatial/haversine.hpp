#pragma once
#include <cudf/cudf.h>

namespace cuSpatial {

/**
 * @param[in] x1: pointer/array of x coordiantes of points

 * @param[in] y1: pointer/array of y coordiantes of points

 * @param[in] x2: pointer/array to index polygons, i.e., prefix-sum of #of rings of all polygons

 * @param[in] y2: pointer/array to index rings, i.e., prefix-sum of #of vertices of all rings

 * @param[out] dist: distance array for each (x1,y1) and (x2,y2) point pari
 */

gdf_column haversine_distance(const gdf_column& x1,const gdf_column& y1,const gdf_column& x2,const gdf_column& y2
                          /* , cudaStream_t stream = 0   */);

}  // namespace cuSpatial

