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
#include <cuspatial/iterator_factory.cuh>

#include <thrust/iterator/transform_iterator.h>

namespace cuspatial {

template <typename PartIterator, typename VecIterator>
struct to_linestring_functor {
  using difference_type = typename thrust::iterator_difference<PartIterator>::type;
  PartIterator part_begin;
  VecIterator point_begin;

  CUSPATIAL_HOST_DEVICE
  to_linestring_functor(PartIterator part_begin, VecIterator point_begin)
    : part_begin(part_begin), point_begin(point_begin)
  {
  }

  CUSPATIAL_HOST_DEVICE auto operator()(difference_type i)
  {
    return linestring_ref{point_begin + part_begin[i], point_begin + part_begin[i + 1]};
  }
};

template <typename PartIterator, typename VecIterator>
class multilinestring_ref;

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE multilinestring_ref<PartIterator, VecIterator>::multilinestring_ref(
  PartIterator part_begin, PartIterator part_end, VecIterator point_begin, VecIterator point_end)
  : _part_begin(part_begin), _part_end(part_end), _point_begin(point_begin), _point_end(point_end)
{
}

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multilinestring_ref<PartIterator, VecIterator>::num_linestrings() const
{
  return thrust::distance(_part_begin, _part_end) - 1;
}

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multilinestring_ref<PartIterator, VecIterator>::part_begin() const
{
  return detail::make_counting_transform_iterator(0,
                                                  to_linestring_functor{_part_begin, _point_begin});
}

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multilinestring_ref<PartIterator, VecIterator>::part_end() const
{
  return part_begin() + num_linestrings();
}

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multilinestring_ref<PartIterator, VecIterator>::point_begin() const
{
  return thrust::next(_point_begin, *_part_begin);
}

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multilinestring_ref<PartIterator, VecIterator>::point_end() const
{
  // _part_end refers to the one past the last part index to the points of this multilinestring.
  // So prior to computing the end point index, we need to decrement _part_end.
  return thrust::next(_point_begin, *thrust::prev(_part_end));
}

template <typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto multilinestring_ref<PartIterator, VecIterator>::operator[](
  IndexType i) const
{
  return *(part_begin() + i);
}

}  // namespace cuspatial
