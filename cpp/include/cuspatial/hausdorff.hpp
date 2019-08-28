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
#include <cudf/cudf.h>

namespace cuspatial {

/**
 * @Brief compute Hausdorff distances among all pairs of a set of trajectories
 * https://en.wikipedia.org/wiki/Hausdorff_distance

 * @param[in] x: x coordinates of the input trajectroies
 * @param[in] y: y coordinates of the input trajectroies
 * @param[in] vertex_counts: numbers of vertices of the set of trajectories;
 * also used to compute the starting offsets of the trjajectories in x/y arrays

 * @returns a flattened (1D) column of all-pairs directed Hausdorff distances among trajectories (i,j)
 * @note Hausdorff distance is not symmetrical
 */

gdf_column directed_hausdorff_distance(const gdf_column& x, const gdf_column& y, const gdf_column& vertex_counts);

}  // namespace cuspatial

