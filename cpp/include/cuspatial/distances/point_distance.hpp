/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/span.hpp>

namespace cuspatial {

/**
 * @ingroup distance
 * @brief Compute pairwise point to point cartesian distance
 *
 * @param points1_x Column of x coordinates to the first point in each pair
 * @param points1_y Column of y coordinates to the first point in each pair
 * @param points2_x Column of x coordinates to the second point in each pair
 * @param points2_y Column of y coordinates to the second point in each pair
 * @param stream The CUDA stream to use for device memory operations and kernel launches
 * @return Column of distances between each pair of input points
 */
std::unique_ptr<cudf::column> pairwise_point_distance(
  cudf::column_view const& points1_x,
  cudf::column_view const& points1_y,
  cudf::column_view const& points2_x,
  cudf::column_view const& points2_y,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial
