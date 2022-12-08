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

#include <optional>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/geometry_collection/multilinestring_ref.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <iterator>

namespace cuspatial {

using namespace detail;

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
struct to_multilinestring_functor {
  using difference_type = typename thrust::iterator_difference<GeometryIterator>::type;
  GeometryIterator _geometry_begin;
  PartIterator _part_begin;
  VecIterator _point_begin;
  VecIterator _point_end;

  CUSPATIAL_HOST_DEVICE
  to_multilinestring_functor(GeometryIterator geometry_begin,
                             PartIterator part_begin,
                             VecIterator point_begin,
                             VecIterator point_end)
    : _geometry_begin(geometry_begin),
      _part_begin(part_begin),
      _point_begin(point_begin),
      _point_end(point_end)
  {
  }

  CUSPATIAL_HOST_DEVICE auto operator()(difference_type i)
  {
    return multilinestring_ref{_part_begin + _geometry_begin[i],
                               thrust::next(_part_begin + _geometry_begin[i + 1]),
                               _point_begin,
                               _point_end};
  }
};

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
class multilinestring_range;

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::multilinestring_range(
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

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::num_multilinestrings()
{
  return thrust::distance(_geometry_begin, _geometry_end) - 1;
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::num_linestrings()
{
  return thrust::distance(_part_begin, _part_end) - 1;
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::num_points()
{
  return thrust::distance(_point_begin, _point_end);
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::multilinestring_begin()
{
  return detail::make_counting_transform_iterator(
    0, to_multilinestring_functor{_geometry_begin, _part_begin, _point_begin, _point_end});
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::multilinestring_end()
{
  return multilinestring_begin() + num_multilinestrings();
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::part_idx_from_point_idx(
  IndexType point_idx)
{
  return thrust::distance(_part_begin, _part_iter_from_point_idx(point_idx));
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE
  thrust::optional<typename thrust::iterator_traits<PartIterator>::difference_type>
  multilinestring_range<GeometryIterator, PartIterator, VecIterator>::part_idx_from_segment_idx(
    IndexType segment_idx)
{
  auto part_idx = thrust::distance(_part_begin, _part_iter_from_point_idx(segment_idx));
  if (not is_valid_segment_id(segment_idx, part_idx)) return thrust::nullopt;
  return part_idx;
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::geometry_idx_from_part_idx(
  IndexType part_idx)
{
  return thrust::distance(_geometry_begin, _geometry_iter_from_part_idx(part_idx));
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::geometry_idx_from_point_idx(
  IndexType point_idx)
{
  return geometry_idx_from_part_idx(part_idx_from_point_idx(point_idx));
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::intra_part_idx(IndexType i)
{
  return i - *_geometry_iter_from_part_idx(i);
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::intra_point_idx(IndexType i)
{
  return i - *_part_iter_from_point_idx(i);
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType1, typename IndexType2>
CUSPATIAL_HOST_DEVICE bool
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::is_valid_segment_id(
  IndexType1 segment_idx, IndexType2 part_idx)
{
  if constexpr (std::is_signed_v<IndexType1>)
    return segment_idx >= 0 && segment_idx < (_part_begin[part_idx + 1] - 1);
  else
    return segment_idx < (_part_begin[part_idx + 1] - 1);
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE thrust::pair<
  vec_2d<typename multilinestring_range<GeometryIterator, PartIterator, VecIterator>::element_t>,
  vec_2d<typename multilinestring_range<GeometryIterator, PartIterator, VecIterator>::element_t>>
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::segment(IndexType segment_idx)
{
  return thrust::make_pair(_point_begin[segment_idx], _point_begin[segment_idx + 1]);
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::operator[](
  IndexType multilinestring_idx)
{
  return multilinestring_begin()[multilinestring_idx];
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::_part_iter_from_point_idx(
  IndexType point_idx)
{
  return thrust::prev(thrust::upper_bound(thrust::seq, _part_begin, _part_end, point_idx));
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::_geometry_iter_from_part_idx(
  IndexType part_idx)
{
  return thrust::prev(thrust::upper_bound(thrust::seq, _geometry_begin, _geometry_end, part_idx));
}

}  // namespace cuspatial
