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
#include <cuspatial/traits.hpp>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/distance.h>

namespace cuspatial {

/**
 * @brief Abstract Data Type (ADT) for any containers representable with a start and end iterator.
 *
 * This is similar to a span, except that the iterators can be composed of generators.
 *
 * @note Although this structure can be used on device and host code, this structure does not
 * provide implicit device-host transfer. It is up to developer's prudence not to access device
 * memory from host or the reverse.
 *
 * @tparam Type of both start and end iterator. IteratorType must statisfy
 * LegacyRandomAccessIterator[LinkLRAI].
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename IteratorType>
class range {
 public:
  using value_type = iterator_value_type<IteratorType>;
  range(IteratorType begin, IteratorType end) : _begin(begin), _end(end) {}

  /// Return the start iterator to the range
  auto CUSPATIAL_HOST_DEVICE begin() { return _begin; }
  /// Return the end iterator to the range
  auto CUSPATIAL_HOST_DEVICE end() { return _end; }
  /// Return the size of the range
  auto CUSPATIAL_HOST_DEVICE size() { return thrust::distance(_begin, _end); }

  /// Access the `i`th element in the range
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
