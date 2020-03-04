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

#include <memory>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

namespace cuspatial {

/**
 * @brief Compute cubic interpolations of a set of points based on their
 * ids and a coefficient matrix.
 *
 * @param[in] query_points column of coordinate values to be interpolated.
 *
 * @param[in] curve_ids int32 column of ids associated with each query point.
 *
 * @param[in] offsets int32 column of offset of the source_points.
 * This is used to calculate which values from the coefficients are
 * used for each interpolation.
 *
 * @param[in] source_points column of the original `t` values used
 * to compute the coefficients matrix.  These source points are used to
 * identify which specific spline a given query_point is interpolated with.
 *
 * @param[in] coefficients table of splines produced by
 * cubicspline_coefficients.
 *
 * @return cudf::column `y` coordinates interpolated from `x` and `coefs`.
**/
std::unique_ptr<cudf::column> cubicspline_interpolate(
                                         cudf::column_view const& query_points,
                                         cudf::column_view const& curve_ids,
                                         cudf::column_view const& offsets,
                                         cudf::column_view const& source_points,
                                         cudf::table_view const& coefficients);

/**
 * @brief Create a table of coefficients from a column of coordinates.
 *
 * Computes coefficients for a natural cubic spline similar to the method
 * found on http://mathworld.wolfram.com/CubicSpline.html .
 *
 * This implementation is massively parallel and has two explicit
 * dependencies: The input data is arranged in parallel arrays and 
 * Structure-of-Array form wherein the values of many functions are
 * packed in sequence into a single array.
 *
 * This library currently requires that all input splines be the same
 * length. The minimum length supported by this algorithm is 5.
 * 
 * @note Ids should be prefixed with a 0, even when only a single spline
 * is fit, ids will be {0, 0}
 *
 * @param[in] t column_view of independent coordinates for fitting splines
 * @param[in] y column_view of dependent variables to be fit along t axis
 * @param[in] ids of incoming coordinate sets
 * @param[in] offsets the index of the end of each coordinate set. Must begin
 * with 0, like the note above for ids.
 *
 * !Important - to enable accelerated performance, this offset is the
 * combined inclusive and exclusive scans and as such should have the same
 * length as ids.
 *
 * For example, in the unit test case, the minimum input set for 3 splines is
 * {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4} but the prefix scans for
 * this input array is {0, 5, 10, 15}
 *
 * @return cudf::table_view of coefficients for spline interpolation. The size
 * of the table is ((M-n), 4) where M is `t.size()` and and n is 
 * `ids.size()-1`.
**/
std::unique_ptr<cudf::experimental::table> cubicspline_coefficients(
                                         cudf::column_view const& t,
                                         cudf::column_view const& y,
                                         cudf::column_view const& ids,
                                         cudf::column_view const& offsets);
}// namespace cuspatial
