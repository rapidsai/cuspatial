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
 * @param[in] points column of coordinate values to be interpolated
 * @param[in] ids column of ids associated with each coordinate value
 * @param[in] coefficients table of splines fit to the curves matching the ids
 * @return cudf::column `y` coordinates interpolated from `x` and `coefs`.
**/
std::unique_ptr<cudf::experimental::column> cubicspline_interpolate(
                                         cudf::column_view points,
                                         cudf::column_view ids,
                                         cudf::table coefficients );

/**
 * @brief Create a table of coefficients from an SoA of coordinates.
 *
 * This version computes coefficients similarly to the table_view method, but
 * only accepts a single column for x. ids and prefix are also passed in as
 * separate arguments.
 *
 * @param[in] x interpolation coordinates for fitting splines
 * @param[in] y column of dependent variables to be fit along x axis
 * @param[in] ids of incoming coordinate sets
 * @param[in] prefix_sum of incoming coordinate sets
 * @return cudf::table_view (4, (M*len(ids))) table of coefficients for spline interpolation
**/
std::unique_ptr<cudf::experimental::table> cubicspline_full(cudf::column_view t,
                                         cudf::column_view y,
                                         cudf::column_view ids,
                                         cudf::column_view prefix_sums);
}// namespace cuspatial
