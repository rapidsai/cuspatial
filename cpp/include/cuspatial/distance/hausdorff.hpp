/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @brief computes Hausdorff distances for all pairs in a collection of spaces
 *
 * @ingroup distance
 *
 * https://en.wikipedia.org/wiki/Hausdorff_distance
 *
 * Example in 1D (this function operates in 2D):
 * ```
 * spaces
 * [0 2 5] [9] [3 7]
 *
 * spaces represented as points per space and concatenation of all points
 * [0 2 5 9 3 7] [3 1 2]
 *
 * note: the following matrices are visually separated to highlight the relationship of a pair of
 * points with the pair of spaces from which it is produced
 *
 * cartesian product of all
 * points by pair of spaces     distance between points
 * +----------+----+-------+    +---------+---+------+
 * : 00 02 05 : 09 : 03 07 :    : 0  2  5 : 9 : 3  7 :
 * : 20 22 25 : 29 : 23 27 :    : 2  0  3 : 7 : 1  5 :
 * : 50 52 55 : 59 : 53 57 :    : 5  3  0 : 4 : 2  2 :
 * +----------+----+-------+    +---------+---+------+
 * : 90 92 95 : 99 : 93 97 :    : 9  7  4 : 0 : 6  2 :
 * +----------+----+-------+    +---------+---+------+
 * : 30 32 35 : 39 : 33 37 :    : 3  1  2 : 6 : 0  4 :
 * : 70 72 75 : 79 : 73 77 :    : 7  5  2 : 2 : 4  0 :
 * +----------+----+-------+    +---------+---+------+

 * minimum distance from
 * every point in one           Hausdorff distance is
 * space to any point in        the maximum of the
 * the other space              minimum distances
 * +----------+----+-------+    +---------+---+------+
 * :  0       :  9 :  3    :    : 0       : 9 : 3    :
 * :     0    :  7 :  1    :    :         :   :      :
 * :        0 :  4 :  2    :    :         :   :      :
 * +----------+----+-------+    +---------+---+------+
 * :        4 :  0 :     2 :    :       4 : 0 :    2 :
 * +----------+----+-------+    +---------+---+------+
 * :     1    :  6 :  0    :    :         : 6 : 0    :
 * :        2 :  2 :     0 :    :       2 :   :      :
 * +----------+----+-------+    +---------+---+------+
 *
 * returned as concatenation of columns
 * [0 2 4 3 0 2 9 6 0]
 * ```
 *
 * @param[in] xs: x component of points
 * @param[in] ys: y component of points
 * @param[in] space_offsets: beginning index of each space, plus the last space's end offset.
 *
 * @returns Hausdorff distances for each pair of spaces
 *
 * @throw cudf::cuda_error if `xs` and `ys` lengths differ
 * @throw cudf::cuda_error if `xs` and `ys` types differ
 * @throw cudf::cuda_error if `space_offsets` size is less than `xs` and `xy`
 * @throw cudf::cuda_error if `xs`, `ys`, or `space_offsets` has nulls
 *
 * @note Hausdorff distances are asymmetrical
 */
std::unique_ptr<cudf::column> directed_hausdorff_distance(
  cudf::column_view const& xs,
  cudf::column_view const& ys,
  cudf::column_view const& space_offsets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
