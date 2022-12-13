
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

#include <cuspatial/traits.hpp>

namespace cuspatial {
namespace detail {

/**
 * @brief Zero set a data range on device
 *
 * @tparam Iterator type of iterator to the range
 * @param begin Start iterator to the range
 * @param end End iterator to the range
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <typename Iterator>
void zero_data_async(Iterator begin, Iterator end, rmm::cuda_stream_view stream)
{
  using value_type = iterator_value_type<Iterator>;
  auto dst         = thrust::raw_pointer_cast(&*begin);
  auto size        = thrust::distance(begin, end) * sizeof(value_type);

  cudaMemsetAsync(dst, 0, size, stream.value());
}

}  // namespace detail
}  // namespace cuspatial
