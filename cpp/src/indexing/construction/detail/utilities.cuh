/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/tuple.h>

namespace cuspatial {
namespace detail {

/**
 * @brief Helper function to reduce verbosity creating cudf fixed-width columns
 */
template <typename T>
inline std::unique_ptr<cudf::column> make_fixed_width_column(
  cudf::size_type size,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<T>()}, size, cudf::mask_state::UNALLOCATED, stream, mr);
}

}  // namespace detail
}  // namespace cuspatial
