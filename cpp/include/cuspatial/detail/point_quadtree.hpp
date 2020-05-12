/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuspatial/point_quadtree.hpp>

namespace cuspatial {
namespace detail {

/**
 * @copydoc cuspatial::quadtree_on_points()
 * @param stream Optional CUDA stream on which to schedule allocations
 */
std::unique_ptr<cudf::experimental::table> quadtree_on_points(
    cudf::mutable_column_view x, cudf::mutable_column_view y, double const x1,
    double const y1, double const x2, double const y2, double const scale,
    cudf::size_type const num_level, cudf::size_type const min_size,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

}  // namespace detail

}  // namespace cuspatial
