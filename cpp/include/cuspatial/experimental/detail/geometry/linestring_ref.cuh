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
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/traits.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace cuspatial {

template <typename VecIterator>
struct to_segment_functor {
  using element_t       = iterator_vec_base_type<VecIterator>;
  using difference_type = typename thrust::iterator_difference<VecIterator>::type;
  VecIterator _point_begin;

  CUSPATIAL_HOST_DEVICE
  to_segment_functor(VecIterator point_begin) : _point_begin(point_begin) {}

  CUSPATIAL_HOST_DEVICE
  segment<element_t> operator()(difference_type i)
  {
    return {_point_begin[i], _point_begin[i + 1]};
  }
};

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE linestring_ref<VecIterator>::linestring_ref(VecIterator begin,
                                                                  VecIterator end)
  : _point_begin(begin), _point_end(end)
{
  using T = iterator_vec_base_type<VecIterator>;
  static_assert(is_same<vec_2d<T>, iterator_value_type<VecIterator>>(), "must be vec2d type");
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto linestring_ref<VecIterator>::num_segments() const
{
  // The number of segment equals the number of points minus 1. And the number of points
  // is thrust::distance(_point_begin, _point_end) - 1.
  return thrust::distance(_point_begin, _point_end) - 1;
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto linestring_ref<VecIterator>::segment_begin() const
{
  return detail::make_counting_transform_iterator(0, to_segment_functor{_point_begin});
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto linestring_ref<VecIterator>::segment_end() const
{
  return segment_begin() + num_segments();
}

template <typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto linestring_ref<VecIterator>::segment(IndexType i) const
{
  return *(segment_begin() + i);
}

}  // namespace cuspatial
