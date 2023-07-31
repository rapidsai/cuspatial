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

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/geometry/linestring_ref.cuh>
#include <cuspatial/geometry/polygon_ref.cuh>
#include <cuspatial/iterator_factory.cuh>

#include <thrust/iterator/transform_iterator.h>

namespace cuspatial {

template <typename PartIterator, typename RingIterator, typename VecIterator>
struct to_polygon_functor {
  using difference_type = typename thrust::iterator_difference<PartIterator>::type;
  PartIterator part_begin;
  RingIterator ring_begin;
  VecIterator point_begin;
  VecIterator point_end;

  CUSPATIAL_HOST_DEVICE
  to_polygon_functor(PartIterator part_begin,
                     RingIterator ring_begin,
                     VecIterator point_begin,
                     VecIterator point_end)
    : part_begin(part_begin), ring_begin(ring_begin), point_begin(point_begin), point_end(point_end)
  {
  }

  CUSPATIAL_HOST_DEVICE auto operator()(difference_type i)
  {
    return polygon_ref{ring_begin + part_begin[i],
                       thrust::next(ring_begin + part_begin[i + 1]),
                       point_begin,
                       point_end};
  }
};

template <typename PartIterator, typename RingIterator, typename VecIterator>
class multipolygon_ref;

template <typename PartIterator, typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE multipolygon_ref<PartIterator, RingIterator, VecIterator>::multipolygon_ref(
  PartIterator part_begin,
  PartIterator part_end,
  RingIterator ring_begin,
  RingIterator ring_end,
  VecIterator point_begin,
  VecIterator point_end)
  : _part_begin(part_begin),
    _part_end(part_end),
    _ring_begin(ring_begin),
    _ring_end(ring_end),
    _point_begin(point_begin),
    _point_end(point_end)
{
}

template <typename PartIterator, typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipolygon_ref<PartIterator, RingIterator, VecIterator>::num_polygons()
  const
{
  return thrust::distance(_part_begin, _part_end) - 1;
}

template <typename PartIterator, typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipolygon_ref<PartIterator, RingIterator, VecIterator>::part_begin()
  const
{
  return detail::make_counting_transform_iterator(
    0, to_polygon_functor{_part_begin, _ring_begin, _point_begin, _point_end});
}

template <typename PartIterator, typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipolygon_ref<PartIterator, RingIterator, VecIterator>::part_end()
  const
{
  return part_begin() + num_polygons();
}

template <typename PartIterator, typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipolygon_ref<PartIterator, RingIterator, VecIterator>::ring_begin()
  const
{
  return detail::make_counting_transform_iterator(0,
                                                  to_linestring_functor{_part_begin, _point_begin});
}

template <typename PartIterator, typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipolygon_ref<PartIterator, RingIterator, VecIterator>::ring_end()
  const
{
  return part_begin() + num_polygons();
}

template <typename PartIterator, typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipolygon_ref<PartIterator, RingIterator, VecIterator>::point_begin()
  const
{
  return thrust::next(_point_begin, *thrust::next(_ring_begin, *_part_begin));
}

template <typename PartIterator, typename RingIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipolygon_ref<PartIterator, RingIterator, VecIterator>::point_end()
  const
{
  return thrust::next(_point_begin, *thrust::next(_ring_begin, *thrust::prev(_part_end)));
}

template <typename PartIterator, typename RingIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto multipolygon_ref<PartIterator, RingIterator, VecIterator>::operator[](
  IndexType i) const
{
  return *(part_begin() + i);
}

}  // namespace cuspatial
