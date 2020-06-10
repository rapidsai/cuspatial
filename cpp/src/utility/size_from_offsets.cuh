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

#include <cudf/column/column_device_view.cuh>

/** @brief Computes a size based on a column of offsets and a final length
 */
struct size_from_offsets_functor {
  cudf::column_device_view offsets;
  cudf::size_type length;

  cudf::size_type __device__ operator()(cudf::size_type idx)
  {
    auto curr_offset = offsets.element<cudf::size_type>(idx);
    auto next_idx    = idx + 1;
    auto next_offset =
      next_idx >= offsets.size() ? length : offsets.element<cudf::size_type>(next_idx);

    return next_offset - curr_offset;
  }
};
