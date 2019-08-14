#pragma once
#include <cudf/cudf.h>

namespace cuSpatial {


gdf_column hausdorff_distance(const gdf_column& coor_x,const gdf_column& coor_y,const gdf_column& traj_cnt
    		/* ,cudaStream_t stream = 0   */);

}  // namespace cuSpatial

