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
 * @brief compute Hausdorff distances among all pairs of a set of trajectories
 * 
 * https://en.wikipedia.org/wiki/Hausdorff_distance
 *
 * `vertex_counts` is used to compute the starting offset of each trajectory
 * in `x` and `y`
 * 
 * @param[in] x: x coordinates of the input trajectories
 * @param[in] y: y coordinates of the input trajectories
 * @param[in] vertex_counts: numbers of vertices in each trajectory
 *
 * @returns Flattened (1D) column of all-pairs directed Hausdorff distances
 *          among trajectories (i,j)
 * 
 * @note Hausdorff distance is not symmetrical
 */
std::unique_ptr<cudf::column>
directed_hausdorff_distance(cudf::column_view const& x,
                            cudf::column_view const& y,
                            cudf::column_view const& vertex_counts,
                            rmm::mr::device_memory_resource *mr =
                              rmm::mr::get_default_resource());

}  // namespace cuspatial
