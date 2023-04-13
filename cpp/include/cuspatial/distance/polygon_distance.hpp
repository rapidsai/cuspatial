/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuspatial/column/geometry_column_view.hpp>

#include <cudf/column/column.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @brief
 *
 * @param lhs
 * @param rhs
 * @param mr
 * @return std::unique_ptr<cudf::column>
 */
std::unique_ptr<cudf::column> pairwise_polygon_distance(
  geometry_column_view const& lhs,
  geometry_column_view const& rhs,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
