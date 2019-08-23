#pragma once
#include <cudf/cudf.h>

namespace cuSpatial {

/**
 * @Brief compute Hausdorff distances among all pairs of a set of trajectories
 * https://en.wikipedia.org/wiki/Hausdorff_distance

 * @param[in] coord_x: x coordinates of the input trajectroies
 * @param[in] coord_y: y coordinates of the input trajectroies
 * @param[in] traj_cnt: numbers of vertices of the set of trajectories;
 * also used to compute the starting offsets of the trjajectories in coord_x/coord_y arrays

 * @returns a flattened (1D) column of all-pairs directed Hausdorff distances among trajectories (i,j)
 * @note Hausdorff distance is not symmetrical
 */

gdf_column directed_hausdorff_distance(const gdf_column& coord_x,const gdf_column& coord_y,const gdf_column& traj_cnt

}  // namespace cuspatial

