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
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/traits.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief Temporary solution to allow structure binding in the range based for loop.
 * To be replaced once cuda::std::tuple is available in libcu++.
 */
template <typename Iterator>
struct to_indexed_pair_functor {
  using value_type = iterator_value_type<Iterator>;

  Iterator _begin;

  CUSPATIAL_HOST_DEVICE
  to_indexed_pair_functor(Iterator begin) : _begin(begin) {}

  template <typename IndexType>
  thrust::pair<IndexType, value_type> CUSPATIAL_HOST_DEVICE operator()(IndexType i)
  {
    return {i, _begin[i]};
  }
};

/**
 * @internal
 * @brief An "enumerated range" is a range that iterate on the element, along with the indices.
 *
 * @tparam Iterator the type of the iterator to the range.
 */
template <typename Iterator>
class enumerate_range {
 public:
  CUSPATIAL_HOST_DEVICE
  enumerate_range(Iterator begin, Iterator end) : _begin(begin), _end(end) {}

  CUSPATIAL_HOST_DEVICE auto begin()
  {
    return make_counting_transform_iterator(0, to_indexed_pair_functor{_begin});
  }
  CUSPATIAL_HOST_DEVICE auto end() { return begin() + thrust::distance(_begin, _end); }

 protected:
  Iterator _begin;
  Iterator _end;
};

}  // namespace detail
}  // namespace cuspatial
