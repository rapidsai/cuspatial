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
namespace cuspatial {


/**
 * @brief pair quadtree quadrants and polygons by intersection tests of quadrants and polygon bboxes
 *
 * @param[in] quadtree: table of five arrays derived from quadtree indexing on points (key,lev,sign, lenght, fpos)
 *
 * @param[in] poly_bbox: table of four arrays representing polygon bboxes (x1,y1,x2,y2)

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

 * @return array of (poygon-idx,quadrant-idx) pairs that quadrant intersects with polygon bbox
 * quadrant-idx and poygon-idx are offsets of quadrant and polygon arrays, respectively
**/

std::unique_ptr<cudf::experimental::table> quad_bbox_join(
    cudf::table_view const& quadtree,cudf::table_view const& poly_bbox,
    double x1,double y1,double x2,double y2, double scale, uint32_t num_level, uint32_t min_size);

/**
 * @brief pair points and polygons using ray-cast based point-in-polygon test algorithm in two phases:
 * phase 1 counts the total number of output pairs for precise memory allocation
 * phase 2 actually writes (point,polygon) pairs
 *
 * @param[in] pq_pair: table of two arrays for (quadrant,polygon) pairs derived from spatial filtering
 *
 * @param[in] quadtree: table of five arrays derived from quadtree indexing on points (key,lev,sign, lenght, fpos)
 *
 * @param[in] pnt: table of two arrays for points (x,y)
 * note that points are in-place sorted in quadtree construction and have different orders than the orginal input points.
 *
 * @param[in] fpos: feature/polygon offset array to rings
 *
 * @param[in] rpos: ring offset array to vertex
 *
 * @param[in] poly_x: polygon x coordiante array.
 *
 * @param[in] poly_y: polygon y coordiante array.
 *
 * @return array of (polygon-idx ,point-idx) pairs that point is within polyon;
 * point-idx and polygon-idx are offsets of point and polygon arrays, respectively
**/

std::unique_ptr<cudf::experimental::table> pip_refine(
    cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
    cudf::column_view const& poly_fpos,cudf::column_view const& poly_rpos,
    cudf::column_view const& poly_x,cudf::column_view const& poly_y);
}// namespace cuspatial
