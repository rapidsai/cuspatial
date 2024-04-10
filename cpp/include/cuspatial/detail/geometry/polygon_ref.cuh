/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/geometry/vec_3d.hpp>
#include <cuspatial/geometry_collection/multilinestring_ref.cuh>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/traits.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace cuspatial {

template <typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE polygon_ref<RingIterator, VecIterator>::polygon_ref(RingIterator ring_begin,
                                                                          RingIterator ring_end,
                                                                          VecIterator point_begin,
                                                                          VecIterator point_end)
  : _ring_begin(ring_begin), _ring_end(ring_end), _point_begin(point_begin), _point_end(point_end)
{
  using T = iterator_vec_base_type<VecIterator>;
  static_assert(is_same<vec_2d<T>, iterator_value_type<VecIterator>>() ||
                  is_same<vec_3d<T>, iterator_value_type<VecIterator>>(),
                "must be vec2d or vec3d type");
}

template <typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto polygon_ref<RingIterator, VecIterator>::num_rings() const
{
  return thrust::distance(_ring_begin, _ring_end) - 1;
}

template <typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto polygon_ref<RingIterator, VecIterator>::ring_begin() const
{
  return detail::make_counting_transform_iterator(0,
                                                  to_linestring_functor{_ring_begin, _point_begin});
}

template <typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto polygon_ref<RingIterator, VecIterator>::ring_end() const
{
  return ring_begin() + size();
}

template <typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto polygon_ref<RingIterator, VecIterator>::point_begin() const
{
  return thrust::next(_point_begin, *_ring_begin);
}

template <typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto polygon_ref<RingIterator, VecIterator>::point_end() const
{
  return thrust::next(_point_begin, *thrust::prev(_ring_end));
}

template <typename RingIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto polygon_ref<RingIterator, VecIterator>::ring(IndexType i) const
{
  return *(ring_begin() + i);
}

}  // namespace cuspatial
