/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>

namespace cuspatial {

namespace detail {

/**
 * @brief For each query point, finds the index of the last coordinate in source_points,
 * that is smaller than the query_point, grouped by prefixes and curve_ids.
 *
 * This implementation of cubic curve has four coefficients for each pair of control points. For
 * a curve then with n control points there will be n-1 sets of coefficients. This function
 * finds which set of coefficients to use for a given query_point. Each `query_point[i]` is passed
 * with a corresponding `curve_ids[i], identifying which offset `j` to use from `prefixes` into
 * `source_points`. For example, given two sets of `source_points` = [0, 1, 2, 3, 4, 0, 2, 5, 10,
 *20] with `prefixes = [0, 5, 10], and a single `query_point = 6` with `curve_ids = 1`, the
 *coefficient position `6` is returned.
 *
 * The first curve, specified by `curve_ids = 0` uses the first four coefficient indices 0...3, and
 * the second curve uses the next four indices. `6 > 5` specifying the third pair (ordinal 2)
 * of `source_points` (also known as control points).
 *
 * Below is a simple diagram of the cofficient indices that correspond with the `source_points`.
 *
 *     [0, 1, 2, 3, 4, 0, 2, 5, 10, 20]
 *       0  1  2  3     4  5  6   7
 *
 * @param query_points column of coordinate values to be interpolated.
 * @param spline_ids ids that identify the spline to interpolate each
 * coordinate into.
 * @param offsets int32 column of offsets of the source_points.
 * This is used to calculate which values from the coefficients are
 * used for each interpolation.
 * @param source_points column of the original `t` values used
 * to compute the coefficients matrix.
 * @param mr the optional caller specified RMM memory resource
 * @param stream the optional caller specified cudaStream
 *
 * @return cudf::column of size equal to query points, one index position
 * of the first source_point mapped by offsets that is smaller than each
 * query point.
 **/
std::unique_ptr<cudf::column> find_coefficient_indices(cudf::column_view const& query_points,
                                                       cudf::column_view const& curve_ids,
                                                       cudf::column_view const& prefixes,
                                                       cudf::column_view const& source_points,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr);

}  // namespace detail

}  // namespace cuspatial
