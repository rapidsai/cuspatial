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

#include "thrust/detail/raw_reference_cast.h"
#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/traits.hpp>

#include <thrust/distance.h>

namespace cuspatial {

template <typename IteratorType>
class range {
 public:
  using value_type = iterator_value_type<IteratorType>;
  range(IteratorType begin, IteratorType end) : _begin(begin), _end(end) {}

  auto CUSPATIAL_HOST_DEVICE begin() { return _begin; }
  auto CUSPATIAL_HOST_DEVICE end() { return _end; }
  auto CUSPATIAL_HOST_DEVICE size() { return thrust::distance(_begin, _end); }

  template <typename IndexType>
  auto& CUSPATIAL_HOST_DEVICE operator[](IndexType i)
  {
    return thrust::raw_reference_cast(_begin[i]);
  }

 private:
  IteratorType _begin;
  IteratorType _end;
};

}  // namespace cuspatial
