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

#include <thrust/distance.h>

namespace cuspatial {

template <typename VecIterator, typename IndexType>
struct point_tile_functor {
  VecIterator points_begin;
  IndexType tile_size;

  CUSPATIAL_HOST_DEVICE auto operator()(IndexType i) { return points_begin[i % tile_size]; }
};
template <typename VecIterator, typename IndexType>
point_tile_functor(VecIterator, IndexType) -> point_tile_functor<VecIterator, IndexType>;

template <typename VecIterator, typename IndexType>
struct point_repeat_functor {
  VecIterator points_begin;
  IndexType repeat_size;

  CUSPATIAL_HOST_DEVICE auto operator()(IndexType i) { return points_begin[i / repeat_size]; }
};
template <typename VecIterator, typename IndexType>
point_repeat_functor(VecIterator, IndexType) -> point_repeat_functor<VecIterator, IndexType>;

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE multipoint_ref<VecIterator>::multipoint_ref(VecIterator begin,
                                                                  VecIterator end)
  : _points_begin(begin), _points_end(end)
{
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_ref<VecIterator>::point_begin() const
{
  return _points_begin;
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_ref<VecIterator>::point_end() const
{
  return _points_end;
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_ref<VecIterator>::num_points() const
{
  return thrust::distance(_points_begin, _points_end);
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_ref<VecIterator>::point_tile_begin() const
{
  return detail::make_counting_transform_iterator(0,
                                                  point_tile_functor{_points_begin, num_points()});
}

template <typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto multipoint_ref<VecIterator>::point_repeat_begin(IndexType repeats) const
{
  return detail::make_counting_transform_iterator(0, point_repeat_functor{_points_begin, repeats});
}

template <typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto multipoint_ref<VecIterator>::operator[](IndexType i)
{
  return point_begin()[i];
}

}  // namespace cuspatial
