/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/types.h>

namespace cuspatial {

/**
 * @brief computes Hausdorff distances for all pairs of a collection of spaces
 * 
 * https://en.wikipedia.org/wiki/Hausdorff_distance
 *
 * `points_per_space` is used to compute the offset of the first point in each space.
 * 
 * @param[in] xs: x coordinate of points in space
 * @param[in] ys: y coordinate of points in space
 * @param[in] points_per_space: number of points in each space
 *
 * @returns A flattened matrix of all Hausdorff distances for each pair of spaces
 * 
 * @note Hausdorff distances are asymmetrical
 */
std::unique_ptr<cudf::column>
directed_hausdorff_distance(cudf::column_view const& xs,
                            cudf::column_view const& ys,
                            cudf::column_view const& points_per_space,
                            rmm::mr::device_memory_resource *mr =
                              rmm::mr::get_default_resource());

}  // namespace cuspatial
