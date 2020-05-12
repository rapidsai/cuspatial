/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuspatial/point_quadtree.hpp>

namespace cuspatial {
namespace detail {

/**
 * @brief construct a quadtree structure from points.
 *
 * @param[in/out] x Column of x coordiantes before[in]/after[out] sorting
 * @param[in/out] y Column of y coordiantes before[in]/after[out] sorting
 * @param[in] x1 X-coordinate of the area of interest bbox's lower left corner
 * @param[in] y1 Y-coordinate of the area of interest bbox's lower left corner
 * @param[in] x2 X-coordinate of the area of interest bbox's upper right corner
 * @param[in] y2 Y-coordinate of the area of interest bbox's upper right corner
 * @param[in] scale Grid cell size along both x and y dimensions. Scale is
 * applied to x1 and x2 to convert x/y coodiantes into a Morton code in 2D
 * space
 * @param[in] num_level Largest depth of quadtree nodes. The value should be
 * less than 16 as uint32_t is used for Morton code representation. The actual
 * number of levels may be less than num_level when # of points are small and/or
 * min_size (next) is large.
 * @param[in] min_size Minimum number of points for a non-leaf quadtree node.
 * All non-last-level quadrants should have less than `min_size` points.
 * Last-level quadrants are permited to have more than `min_size` points.
 * `min_size` is typically set to the number of threads in a block used in
 * the two CUDA kernels needed in the spatial refinement step.
 * @param[in] mr The optional resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to schedule allocations
 *
 * @return cuDF table with five columns for a complete quadtree:
 * key, lev, sign, length, fpos
 * see http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf and other
 * docs for details.
 */
std::unique_ptr<cudf::experimental::table> quadtree_on_points(
    cudf::mutable_column_view x, cudf::mutable_column_view y, double const x1,
    double const y1, double const x2, double const y2, double const scale,
    cudf::size_type const num_level, cudf::size_type const min_size,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

}  // namespace detail

}  // namespace cuspatial
