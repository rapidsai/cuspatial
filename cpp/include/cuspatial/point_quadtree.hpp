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
#include <utility/quadtree_thrust.cuh>
namespace cuspatial {

/**
 * @brief construct a quadtree structure from points.
 *
 * @param[in/out] x: column of x coordiantes before[in]/after[out] sorting.
 *
 * @param[in/out] y: column of y coordiantes before[in]/after[out] sorting.

 * @param[in] x1/y1/x2/y2: bounding box of area of interests.

 * @param[in] scale: grid cell size along both x and y dimensions.
 * scale works with x1 and x2 to convert x/y coodiantes into a Morton code in 2D space

 *@ param[in] num_level: largest depth of quadtree nodes
 * the value should be less than 16 as uint32_t is used for Morton code representation
 * the actual number of levels may be less than num_level
 * when #of points are small and/or min_size (next) is large

 *@ param[in] min_size: the minimum number of points for a non-leaf quadtree node
 *  all non-last-level quadrants should have less than min_size points
 *  last-level quadrants are permited to have more than min_size points
 *  min_size is typically set to the number of threads in a block used in
 *  the two CUDA kernels needed in the spatial refinment step.

 * @return experimental::table with five columns for a complete quadtree: key,lev,sign,length, fpos
 * see http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf and other docs for details.

**/
std::unique_ptr<cudf::experimental::table> quadtree_on_points(
    cudf::mutable_column_view& x,cudf::mutable_column_view& y,
    double x1,double y1,double x2,double y2,
    double scale, int num_level, int min_size);

}// namespace cuspatial
