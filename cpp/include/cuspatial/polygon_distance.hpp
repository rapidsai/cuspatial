/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/types.hpp>

#include <memory>

namespace cuspatial {

/**
 * @brief
 *
 * @param xs
 * @param ys
 * @param shape_offsets
 * @param mr
 * @return std::unique_ptr<cudf::column>
 */
std::unique_ptr<cudf::column> directed_polygon_distance(
  cudf::column_view const& xs,
  cudf::column_view const& ys,
  cudf::column_view const& shape_offsets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
