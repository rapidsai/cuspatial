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



/**
 * @Brief deriving trajectories from points (x/y relative to an origin), timestamps and objectids
 * by first sorting based on id and timestamp and then group by id.

 * @param[in/out] coor_x/coor_y: x and y coordiantes reative to a camera origin (before/after sorting)

 * @param[in/out] oid: object (e.g., vehicle) id column (before/after sorting); upon completion, unqiue ids become trajectory ids

 * @param[in/out] ts: timestamp column (before/after sorting)

 * @param[out] tid: trajectory id column (see comments on oid)

 * @param[out] len: #of points in the derived trajectories

 * @param[out] pos: position offsets of trajectories used to index coor_x/coor_y/oid/ts after completion

 * @returns the number of derived trajectory

 */
int coor2traj(gdf_column& coor_x,gdf_column& coor_y,gdf_column& oid, gdf_column& ts,
 			      gdf_column& tid, gdf_column& len,gdf_column& pos/* ,cudaStream_t stream = 0   */);


/**
 * @Brief computing distance(length) and speed of trajectories after their formation (e.g., from coor2traj)

 * @param[in] coor_x/coor_y: x and y coordiantes reative to a camera origin ordered by (id,timestamp)

 * @param[in] ts: timestamp column ordered by (id,timestamp)

 * @param[in] len: number of points column ordered by (id,timestamp)

 * @param[in] pos: position offsets of trajectories used to index coor_x/coor_y/oid/ts ordered by (id,timestamp)

 * @param[out] dist: computed distances/lengths of trajectories in meters (m)

 * @param[out] speed: computed speed of trajectories in meters per second (m/s)

 * Note: we might output duration (in addtiion to distance/speed), should there is a need
 * duration can be easiy computed on CPUs by fetching begining/ending timestamps of a trajectory in the timestamp array
 */

void traj_distspeed(const gdf_column& coor_x,const gdf_column& coor_y,const gdf_column& ts,
 			    const gdf_column& len,const gdf_column& pos,gdf_column& dist,gdf_column& speed
 			    /* ,cudaStream_t stream = 0   */);



/**
 * @Brief computing spatial bounding boxes of trjectories

 * @param[in] coor_x/coor_y: x and y coordiantes reative to a camera origin ordered by (id,timestamp)

 * @param[in] len: number of points column ordered by (id,timestamp)

 * @param[in] pos: position offsets of trajectories used to index coor_x/coor_y/ ordered by (id,timestamp)

 * @param[out] bbox_x1/bbox_y1/bbox_x2/bbox_y2: computed spaital bounding boxes in four columns

 * Note: temporal 1D bounding box can be computed similary but it seems that there is no such a need;
 * Similar to the dicussion in coor2traj, the temporal 1D bounding box can be retrieved directly
 */

void traj_sbbox(const gdf_column& coor_x,const gdf_column& coor_y,
 			    const gdf_column& len,const gdf_column& pos,
				gdf_column& bbox_x1,gdf_column& bbox_y1,gdf_column& bbox_x2,gdf_column& bbox_y2
 			    	/* ,cudaStream_t stream = 0   */);
}  // namespace cuSpatial
