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

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup cubic_spline
 * @{
 */

/**
 * @brief Create a table of cubic spline coefficients from columns of coordinates.
 *
 * Computes coefficients for a natural cubic spline similar to the method
 * found on http://mathworld.wolfram.com/CubicSpline.html .
 *
 * The input data arrays `t` and `y` contain the vertices of many concatenated
 * splines.
 *
 * Currently, all input splines must be the same length. The minimum supported
 * length is 5.
 *
 * @note Ids should be prefixed with a 0, even when only a single spline
 * is fit, ids will be {0, 0}
 *
 * @param[in] t column_view of independent coordinates for fitting splines
 * @param[in] y column_view of dependent variables to be fit along t axis
 * @param[in] ids of incoming coordinate sets
 * @param[in] offsets the exclusive scan of the spline sizes, prefixed by
 * 0. For example, for 3 splines of 5 vertices each, the offsets input array
 * is {0, 5, 10, 15}.
 * @param[in] mr The memory resource to use for allocating output
 *
 * @return cudf::table_view of coefficients for spline interpolation. The size
 * of the table is ((M-n), 4) where M is `t.size()` and and n is
 * `ids.size()-1`.
 **/
std::unique_ptr<cudf::column> cubicspline_interpolate(
  cudf::column_view const& query_points,
  cudf::column_view const& spline_ids,
  cudf::column_view const& offsets,
  cudf::column_view const& source_points,
  cudf::table_view const& coefficients,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute cubic interpolations of a set of points based on their
 * ids and a coefficient matrix.
 *
 * @param[in] query_points column of coordinate values to be interpolated.
 * @param[in] spline_ids ids that identift the spline to interpolate each
 * coordinate into.
 * @param[in] offsets int32 column of offset of the source_points.
 * This is used to calculate which values from the coefficients are
 * used for each interpolation.
 * @param[in] source_points column of the original `t` values used
 * to compute the coefficients matrix.  These source points are used to
 * identify which specific spline a given query_point is interpolated with.
 * @param[in] coefficients table of spline coefficients produced by
 * cubicspline_coefficients.
 * @param[in] mr The memory resource to use for allocating output
 *
 * @return cudf::column `y` coordinates interpolated from `x` and `coefs`.
 **/
std::unique_ptr<cudf::table> cubicspline_coefficients(
  cudf::column_view const& t,
  cudf::column_view const& y,
  cudf::column_view const& ids,
  cudf::column_view const& offsets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */
}  // namespace cuspatial
