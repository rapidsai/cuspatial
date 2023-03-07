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

#pragma once

#include <thrust/binary_search.h>
#include <thrust/distance.h>

/** @brief Given list offset and row `i`, return a unique key that represent the list of `i`.
 *
 *  The key is computed by performing a `upper_bound` search with `i` in the offset array.
 *  Then subtracts the position with the start of offset array.
 *
 *  Example:
 *  offset:  0 0 0 1 3 4 4 4
 *  i:       0 1 2 3
 *  key:     3 4 4 5
 *
 *  Note that the values of `key`, {offset[3], offset[4], offset[5]} denotes the ending
 *  position of the first 3 non-empty list.
 */
template <typename Iterator>
struct offsets_to_keys_functor {
  Iterator _offsets_begin;
  Iterator _offsets_end;

  offsets_to_keys_functor(Iterator offset_begin, Iterator offset_end)
    : _offsets_begin(offset_begin), _offsets_end(offset_end)
  {
  }

  template <typename IndexType>
  IndexType __device__ operator()(IndexType i)
  {
    return thrust::distance(_offsets_begin,
                            thrust::upper_bound(thrust::seq, _offsets_begin, _offsets_end, i));
  }
};
