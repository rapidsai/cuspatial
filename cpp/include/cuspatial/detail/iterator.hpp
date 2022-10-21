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
#include <cuspatial/cuda_utils.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cuspatial {
namespace detail {

template <typename IndexType, typename UnaryFunction>
inline CUSPATIAL_HOST_DEVICE auto make_counting_transform_iterator(IndexType start, UnaryFunction f)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator(start), f);
}
}  // namespace detail
}  // namespace cuspatial
