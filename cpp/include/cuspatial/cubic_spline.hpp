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
 * @brief Compute quadratic interpolations of a set of points based on their
 * ids and a coefficient matrix.
 *
 * @param[in] query_points float32 column of coordinate values to be 
 * interpolated.
 *
 * @param[in] curve_ids int32 column of ids associated with each query point.
 * These ids identify which spline each query point should be interpolated
 * into.
 *
 * @param[in] prefixes int32 column of prefix_sum of the source_points.
 * This is used to calculate which values from the coefficients are
 * used for each interpolation.
 *
 * @param[in] source_points float32 column of the original `t` values used
 * to compute the coefficients matrix.  These source points are used to
 * identify which specific spline a given query_point is interpolated with.
 *
 * @param[in] coefficients float32 table of splines produced by
 * cubicspline_coefficients.
 *
 * @return cudf::column `y` coordinates interpolated from `x` and `coefs`.
**/
std::unique_ptr<cudf::column> cubicspline_interpolate(
                                         cudf::column_view const& query_points,
                                         cudf::column_view const& curve_ids,
                                         cudf::column_view const& prefixes,
                                         cudf::column_view const& source_points,
                                         cudf::table_view const& coefficients);

/**
 * @brief Create a table of coefficients from an SoA of coordinates.
 *
 * Computes coefficients for a natural cubic spline according to the method
 * found on http://mathworld.wolfram.com/CubicSpline.html with some
 * variation. 
 *
 * This implementation is massively parallel and has two explicit
 * dependencies: The input data is arranged in parallel arrays and 
 * Structure-of-Array form wherein the values of many functions are
 * packed in sequence into a single array.
 *
 * This library currently requires that all input splines be the same
 * length. The minimum length supported by this algorithm is 5.
 *
 * @param[in] float32 column_view of x interpolation coordinates for fitting splines
 * @param[in] float32 y column of dependent variables to be fit along x axis
 * @param[in] int32 ids of incoming coordinate sets
 *
 * !Important - to enable accelerated performance, ids should be prefixed
 * with a 0, even when only a single spline is fit, ids will be {0, 0}
 * 
 * @param[in] int32 prefix_sum of incoming coordinate sets
 *
 * !Important - to enable accelerated performance, this prefix_sum is the
 * combined inclusive and exclusive scans and as such should have the same
 * length as ids.
 *
 * For example, in the unit test case, the minimum input set for 3 splines is
 * {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4} but the prefix scans for
 * this input array is {0, 5, 10, 15}
 *
 * @return cudf::table_view (4, (M*len(ids))) table of coefficients for spline interpolation where M is (len(ids)-1).
**/
std::unique_ptr<cudf::experimental::table> cubicspline_coefficients(
                                         cudf::column_view const& t,
                                         cudf::column_view const& y,
                                         cudf::column_view const& ids,
                                         cudf::column_view const& prefix_sums);
}// namespace cuspatial
