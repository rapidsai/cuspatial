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
#include <cuspatial/experimental/geometry_collection/multipoint_ref.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

namespace cuspatial {

using namespace cuspatial::detail;

namespace detail {
template <typename GeometryIterator, typename VecIterator>
struct to_multipoint_functor {
  using difference_type = typename thrust::iterator_difference<GeometryIterator>::type;
  GeometryIterator _offset_iter;
  VecIterator _points_begin;

  to_multipoint_functor(GeometryIterator offset_iter, VecIterator points_begin)
    : _offset_iter(offset_iter), _points_begin(points_begin)
  {
  }

  CUSPATIAL_HOST_DEVICE
  auto operator()(difference_type const& i)
  {
    return multipoint_ref<VecIterator>{_points_begin + _offset_iter[i],
                                       _points_begin + _offset_iter[i + 1]};
  }
};

}  // namespace detail

template <typename GeometryIterator, typename VecIterator>
multipoint_range<GeometryIterator, VecIterator>::multipoint_range(GeometryIterator geometry_begin,
                                                                  GeometryIterator geometry_end,
                                                                  VecIterator points_begin,
                                                                  VecIterator points_end)
  : _geometry_begin(geometry_begin),
    _geometry_end(geometry_end),
    _points_begin(points_begin),
    _points_end(points_end)
{
}

template <typename GeometryIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_range<GeometryIterator, VecIterator>::num_multipoints()
{
  return thrust::distance(_geometry_begin, _geometry_end) - 1;
}

template <typename GeometryIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_range<GeometryIterator, VecIterator>::num_points()
{
  return thrust::distance(_points_begin, _points_end);
}

template <typename GeometryIterator, typename VecIterator>
auto multipoint_range<GeometryIterator, VecIterator>::multipoint_begin()
{
  return cuspatial::detail::make_counting_transform_iterator(
    0, detail::to_multipoint_functor(_geometry_begin, _points_begin));
}

template <typename GeometryIterator, typename VecIterator>
auto multipoint_range<GeometryIterator, VecIterator>::multipoint_end()
{
  return multipoint_begin() + size();
}

template <typename GeometryIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_range<GeometryIterator, VecIterator>::point_begin()
{
  return _points_begin;
}

template <typename GeometryIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_range<GeometryIterator, VecIterator>::point_end()
{
  return _points_end;
}

template <typename GeometryIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_range<GeometryIterator, VecIterator>::offsets_begin()
{
  return _geometry_begin;
}

template <typename GeometryIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_range<GeometryIterator, VecIterator>::offsets_end()
{
  return _geometry_end;
}

template <typename GeometryIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto multipoint_range<GeometryIterator, VecIterator>::operator[](
  IndexType idx)
{
  return multipoint_ref<VecIterator>{_points_begin + _geometry_begin[idx],
                                     _points_begin + _geometry_begin[idx + 1]};
}

template <typename GeometryIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multipoint_range<GeometryIterator, VecIterator>::geometry_idx_from_point_idx(IndexType idx) const
{
  return thrust::distance(
    _geometry_begin,
    thrust::prev(thrust::upper_bound(thrust::seq, _geometry_begin, _geometry_end, idx)));
}

}  // namespace cuspatial
