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
#include <cuspatial/detail/functors.cuh>
#include <cuspatial/detail/multilinestring_segment.cuh>
#include <cuspatial/detail/utility/validation.hpp>
#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/geometry_collection/multipolygon_ref.cuh>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/multilinestring_range.cuh>
#include <cuspatial/range/multipoint_range.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

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
  RingIterator _ring_end;
  VecIterator _point_begin;
  VecIterator _point_end;

  CUSPATIAL_HOST_DEVICE
  to_multipolygon_functor(GeometryIterator geometry_begin,
                          PartIterator part_begin,
                          RingIterator ring_begin,
                          RingIterator ring_end,
                          VecIterator point_begin,
                          VecIterator point_end)
    : _geometry_begin(geometry_begin),
      _part_begin(part_begin),
      _ring_begin(ring_begin),
      _ring_end(ring_end),
      _point_begin(point_begin),
      _point_end(point_end)
  {
  }

  CUSPATIAL_HOST_DEVICE auto operator()(difference_type i)
  {
    return multipolygon_ref{_part_begin + _geometry_begin[i],
                            thrust::next(_part_begin, _geometry_begin[i + 1] + 1),
                            _ring_begin,
                            _ring_end,
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
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::multipolygon_range(
  GeometryIterator geometry_begin,
  GeometryIterator geometry_end,
  PartIterator part_begin,
  PartIterator part_end,
  RingIterator ring_begin,
  RingIterator ring_end,
  VecIterator point_begin,
  VecIterator point_end)
  : _geometry_begin(geometry_begin),
    _geometry_end(geometry_end),
    _part_begin(part_begin),
    _part_end(part_end),
    _ring_begin(ring_begin),
    _ring_end(ring_end),
    _point_begin(point_begin),
    _point_end(point_end)
{
  static_assert(
    is_vec_2d<iterator_value_type<VecIterator>> || is_vec_3d<iterator_value_type<VecIterator>>,
    "point_begin and point_end must be iterators to floating point vec_2d types or vec_3d.");

  CUSPATIAL_EXPECTS_VALID_MULTIPOLYGON_SIZES(
    num_points(), num_multipolygons() + 1, num_polygons() + 1, num_rings() + 1);
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::num_multipolygons()
{
  return thrust::distance(_geometry_begin, _geometry_end) - 1;
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::num_polygons()
{
  return thrust::distance(_part_begin, _part_end) - 1;
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::num_rings()
{
  return thrust::distance(_ring_begin, _ring_end) - 1;
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::num_points()
{
  return thrust::distance(_point_begin, _point_end);
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::multipolygon_begin()
{
  return detail::make_counting_transform_iterator(
    0,
    to_multipolygon_functor{
      _geometry_begin, _part_begin, _ring_begin, _ring_end, _point_begin, _point_end});
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::multipolygon_end()
{
  return multipolygon_begin() + num_multipolygons();
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::point_begin()
{
  return _point_begin;
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::point_end()
{
  return _point_end;
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::
  ring_idx_from_point_idx(IndexType point_idx)
{
  return thrust::distance(
    _ring_begin, thrust::prev(thrust::upper_bound(thrust::seq, _ring_begin, _ring_end, point_idx)));
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::
  part_idx_from_ring_idx(IndexType ring_idx)
{
  return thrust::distance(
    _part_begin, thrust::prev(thrust::upper_bound(thrust::seq, _part_begin, _part_end, ring_idx)));
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::
  geometry_idx_from_part_idx(IndexType part_idx)
{
  return thrust::distance(
    _geometry_begin,
    thrust::prev(thrust::upper_bound(thrust::seq, _geometry_begin, _geometry_end, part_idx)));
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::
  multipolygon_point_count_begin()
{
  auto multipolygon_point_offset_it = thrust::make_permutation_iterator(
    _ring_begin, thrust::make_permutation_iterator(_part_begin, _geometry_begin));

  auto point_offset_pair_it = thrust::make_zip_iterator(multipolygon_point_offset_it,
                                                        thrust::next(multipolygon_point_offset_it));

  return thrust::make_transform_iterator(point_offset_pair_it,
                                         detail::offset_pair_to_count_functor{});
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::
  multipolygon_point_count_end()
{
  return multipolygon_point_count_begin() + num_multipolygons();
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::
  multipolygon_ring_count_begin()
{
  auto multipolygon_ring_offset_it =
    thrust::make_permutation_iterator(_part_begin, _geometry_begin);

  auto ring_offset_pair_it = thrust::make_zip_iterator(multipolygon_ring_offset_it,
                                                       thrust::next(multipolygon_ring_offset_it));

  return thrust::make_transform_iterator(ring_offset_pair_it,
                                         detail::offset_pair_to_count_functor{});
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::
  multipolygon_ring_count_end()
{
  return multipolygon_ring_count_begin() + num_multipolygons();
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType1, typename IndexType2>
CUSPATIAL_HOST_DEVICE bool
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::is_valid_segment_id(
  IndexType1 point_idx, IndexType2 ring_idx)
{
  if constexpr (std::is_signed_v<IndexType1>)
    return point_idx >= 0 && point_idx < (_ring_begin[ring_idx + 1] - 1);
  else
    return point_idx < (_ring_begin[ring_idx + 1] - 1);
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::operator[](
  IndexType multipolygon_idx)
{
  return multipolygon_begin()[multipolygon_idx];
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
auto multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::_segments(
  rmm::cuda_stream_view stream)
{
  auto multilinestring_range = this->as_multilinestring_range();
  return multilinestring_segment_manager{multilinestring_range, stream};
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::as_multipoint_range()
{
  auto multipoint_geometry_it = thrust::make_permutation_iterator(
    _ring_begin, thrust::make_permutation_iterator(_part_begin, _geometry_begin));

  return multipoint_range{multipoint_geometry_it,
                          multipoint_geometry_it + thrust::distance(_geometry_begin, _geometry_end),
                          _point_begin,
                          _point_end};
}

template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multipolygon_range<GeometryIterator, PartIterator, RingIterator, VecIterator>::
  as_multilinestring_range()
{
  auto multilinestring_geometry_it =
    thrust::make_permutation_iterator(_part_begin, _geometry_begin);
  return multilinestring_range{
    multilinestring_geometry_it,
    multilinestring_geometry_it + thrust::distance(_geometry_begin, _geometry_end),
    _ring_begin,
    _ring_end,
    _point_begin,
    _point_end};
}

}  // namespace cuspatial
