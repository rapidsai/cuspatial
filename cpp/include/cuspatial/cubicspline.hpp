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

std::unique_ptr<cudf::experimental::table> cubicspline(cudf::column_view t,
                                         cudf::column_view x,
                                         cudf::column_view ids);

/**
 * @brief Create a table of coefficients from an SoA of x/y coordinates
 *
 * @param[in] x interpolation coordinates for fitting splines
 * @param[in] y table of depedent variables to be fit along x axis
 * @param[in] ids_and_end_coordinates pairs of ids and index of last value in each trajectory
 * @return cudf::table_view (4, (M*len(ids))) table of coefficients for spline interpolation
**/
std::unique_ptr<cudf::experimental::table> cubicspline(cudf::column_view x,
                                         cudf::table_view y,
                                         cudf::table_view ids);
}// namespace cuspatial
