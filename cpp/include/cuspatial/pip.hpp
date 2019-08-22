#pragma once
#include <cudf/cudf.h>

namespace cuSpatial {

/**
 * @Brief Point-in-Polygon (PIP) tests among a vector/array of points and a vector/array of polygons

 * @param[in] pnt_x: pointer/array of x coordiantes of points
 * @param[in] pnt_y: pointer/array of y coordiantes of points
 * @param[in] ply_fpos: pointer/array to index polygons, i.e., prefix-sum of #of rings of all polygons
 * @param[in] ply_rpos: pointer/array to index rings, i.e., prefix-sum of #of vertices of all rings
 * @param[in] ply_x: pointer/array of x coordiantes of concatenated polygons
 * @param[in] ply_y: pointer/array of x coordiantes of concatenated polygons
 *
 * @returns gdf_column of type GDF_INT32; the jth bit of the ith element of the returned GDF_INT32 array indicate
 * whether the ith point is in the jth polygon.

 * Note: The # of polygons, i.e., ply_fpos.size can not exceed sizeof(uint)*8, i.e., 32.
 */

gdf_column pip_bm(const gdf_column& pnt_x,const gdf_column& pnt_y,
                                   const gdf_column& ply_fpos, const gdf_column& ply_rpos,
                                   const gdf_column& ply_x,const gdf_column& ply_y
                          /* , cudaStream_t stream = 0   */);

}  // namespace cuSpatial

