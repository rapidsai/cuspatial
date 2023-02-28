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

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/geometry_collection/multipolygon_ref.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <iterator>
#include <optional>

namespace cuspatial {

using namespace detail;

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
struct to_multipolygon_functor {
  using difference_type = typename thrust::iterator_difference<GeometryIterator>::type;
  GeometryIterator _geometry_begin;
  PartIterator _part_begin;
  RingIterator _ring_begin;
  VecIterator _point_begin;
  VecIterator _point_end;

  CUSPATIAL_HOST_DEVICE
  to_multipolygon_functor(GeometryIterator geometry_begin,
                          PartIterator part_begin,
                          RingIterator ring_begin,
                          VecIterator point_begin,
                          VecIterator point_end)
    : _geometry_begin(geometry_begin),
      _part_begin(part_begin),
      _ring_begin(ring_begin),
      _point_begin(point_begin),
      _point_end(point_end)
  {
  }

  CUSPATIAL_HOST_DEVICE auto operator()(difference_type i)
  {
    return multipolygon_ref{_part_begin + _geometry_begin[i],
                            thrust::next(_part_begin + _geometry_begin[i + 1]),
                            _point_begin,
                            _point_end};
  }
};

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
class multipolygon_range;

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::multipolygon_range(
  GeometryIterator geometry_begin,
  GeometryIterator geometry_end,
  PartIterator part_begin,
  PartIterator part_end,
  VecIterator point_begin,
  VecIterator point_end)
  : _geometry_begin(geometry_begin),
    _geometry_end(geometry_end),
    _part_begin(part_begin),
    _part_end(part_end),
    _point_begin(point_begin),
    _point_end(point_end)
{
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::num_multipolygons()
{
  return thrust::distance(_geometry_begin, _geometry_end) - 1;
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::num_polygons()
{
  return thrust::distance(_part_begin, _part_end) - 1;
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::num_rings()
{
  return thrust::distance(_ring_begin, _ring_end) - 1;
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::num_points()
{
  return thrust::distance(_point_begin, _point_end);
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::multipolygon_begin()
{
  return detail::make_counting_transform_iterator(
    0,
    to_multipolygon_functor{_geometry_begin, _part_begin, _ring_begin, _point_begin, _point_end});
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::multipolygon_end()
{
  return multipolygon_begin() + num_multipolygons();
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::ring_idx_from_point_idx(
  IndexType point_idx)
{
  return thrust::distance(_ring_begin,
                          thrust::prev(thrust::upper_bound(_ring_begin, _ring_end, point_idx)));
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::part_idx_from_ring_idx(
  IndexType ring_idx)
{
  return thrust::distance(_part_begin,
                          thrust::prev(thrust::upper_bound(_part_begin, _part_begin, ring_idx)));
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::geometry_idx_from_part_idx(
  IndexType part_idx)
{
  return thrust::distance(
    _geometry_begin, thrust::prev(thrust::upper_bound(_geometry_begin, _geometry_end, part_idx)));
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::geometry_idx_from_point_idx(
  IndexType point_idx)
{
  return geometry_idx_from_part_idx(part_idx_from_ring_idx(ring_idx_from_part_idx(point_idx)));
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, VecIterator>::operator[](
  IndexType multipolygon_idx)
{
  return multipolygon_begin()[multipolygon_idx];
}

}  // namespace cuspatial
