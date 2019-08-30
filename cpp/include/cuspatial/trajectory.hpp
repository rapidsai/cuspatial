/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/cudf.h>

namespace cuspatial {

/**
 * @brief derive trajectories from points, timestamps and object ids
 * 
 * Points are x/y coordinates relative to an origin). First sorts based on 
 * object id and timestamp and then groups by id.
 * 
 * @param[in/out] x: x coordinates relative to a camera origin
 *                  (before/after sorting)
 * @param[in/out] y: y coordinates relative to a camera origin
 *                   (before/after sorting)
 * @param[in/out] object_id: object (e.g., vehicle) id column (before/after 
 *                sorting); upon completion, unique ids become trajectory ids
 * @param[in/out] ts: timestamp column (before/after sorting)
 * @param[out] tid: trajectory id column (see comments on oid)
 * @param[out] len: #of points in the derived trajectories
 * @param[out] pos: position offsets of trajectories used to index x, y, 
 *                  object_id and timestamp
 * @returns the number of derived trajectories
 */
int coords_to_trajectories(gdf_column& x, gdf_column& y, gdf_column& object_id,
                           gdf_column& timestamp, gdf_column& trajectory_id,
                           gdf_column& len, gdf_column& pos);


/**
 * @brief Compute the distance and speed of trajectories
 *
 * Trajectories are typically derived from coordinate data using coords_to_trajectories).
 * @param[in] x: x coordinates relative to a camera origin and ordered by (id,timestamp)
 * @param[in] y: y coordinates relative to a camera origin and ordered by (id,timestamp)
 * @param[in] ts: timestamp column ordered by (id,timestamp)
 * @param[in] len: number of points column ordered by (id,timestamp)
 * @param[in] pos: position offsets of trajectories used to index x/y/oid/ts ordered by (id,timestamp)
 * @param[out] dist: computed distances/lengths of trajectories in meters (m)
 * @param[out] speed: computed speed of trajectories in meters per second (m/s)

 * Note: we might output duration (in addition to distance/speed), should there is a need
 * duration can be easily computed on CPUs by fetching begining/ending timestamps of a trajectory in the timestamp array
 */
std::pair<gdf_column,gdf_column> trajectory_distance_and_speed(const gdf_column& x,
                                                               const gdf_column& y,
                                                               const gdf_column& ts,
                                                               const gdf_column& len,
                                                               const gdf_column& pos);


/**
 * @brief compute spatial bounding boxes of trajectories

 * @param[in] x: x coordinates relative to a camera origin and ordered by (id,timestamp)
 * @param[in] y: y coordinates relative to a camera origin and ordered by (id,timestamp)
 * @param[in] len: number of points column ordered by (id,timestamp)
 * @param[in] pos: position offsets of trajectories used to index x/y/ ordered by (id,timestamp)
 * @param[out] bbox_x1: x coordinates of the lower-left corners of computed spatial bounding boxes
 * @param[out] bbox_y1: y coordinates of the lower-left corners of computed spatial bounding boxes
 * @param[out] bbox_x2: x coordinates of the upper-right corners of computed spatial bounding boxes
 * @param[out] bbox_y2: y coordinates of the upper-right corners of computed spatial bounding boxes

 * Note: temporal 1D bounding box can be computed similary but it seems that there is no such a need;
 * Similar to the discussion in coord_to_traj, temporal 1D bounding box can be retrieved directly
 */
void trajectory_spatial_bounds(const gdf_column& x, const gdf_column& y,
                               const gdf_column& len, const gdf_column& pos,
                               gdf_column& bbox_x1, gdf_column& bbox_y1,
                               gdf_column& bbox_x2, gdf_column& bbox_y2);

}  // namespace cuspatial