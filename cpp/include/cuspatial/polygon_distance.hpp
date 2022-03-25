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

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

namespace cuspatial {

/**
 * @brief Compute the shortest distance between every pair of spaces.
 *
 * @param [in] xs The x-coordinates of input points
 * @param [in] ys The y-coordinates of input points
 * @param [in] space_offsets The indices that marks the start of each space, plus the end index.
 * @param [in] mr Memory resource that's used to allocate result column.
 *
 * @return A cudf::column that contains the shortest distances
 */

std::unique_ptr<cudf::column> polygon_distance(
  cudf::column_view const& xs,
  cudf::column_view const& ys,
  cudf::device_span<cudf::size_type> const& space_offsets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
