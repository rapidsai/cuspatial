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

#include <iterator>

namespace cuspatial {

using namespace cuspatial::detail;

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
class multilinestring_range;

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::multilinestring_range(
  GeometryIterator geometry_begin,
  GeometryIterator geometry_end,
  PartIterator part_begin,
  PartIterator part_end,
  VecIterator points_begin,
  VecIterator points_end)
  : geometry_begin(geometry_begin),
    geometry_end(geometry_end),
    part_begin(part_begin),
    part_end(part_end),
    points_begin(points_begin),
    points_end(points_end)
{
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::size()
{
  return num_multilinestrings();
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::num_multilinestrings()
{
  return thrust::distance(geometry_begin, geometry_end) - 1;
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::num_linestrings()
{
  return thrust::distance(part_begin, part_end) - 1;
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::num_points()
{
  return thrust::distance(points_begin, points_end);
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::part_idx_from_point_idx(
  IndexType point_idx)
{
  auto part_it = thrust::upper_bound(thrust::seq, part_begin, part_end, point_idx);
  return thrust::distance(part_begin, thrust::prev(part_it));
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::geometry_idx_from_part_idx(
  IndexType part_idx)
{
  auto geom_it = thrust::upper_bound(thrust::seq, geometry_begin, geometry_end, part_idx);
  return thrust::distance(geometry_begin, thrust::prev(geom_it));
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
template <typename IndexType1, typename IndexType2>
CUSPATIAL_HOST_DEVICE bool
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::is_valid_segment_id(
  IndexType1 segment_idx, IndexType2 part_idx)
{
  return segment_idx >= 0 && segment_idx < (part_begin[part_idx + 1] - 1);
}

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE thrust::pair<
  vec_2d<typename multilinestring_range<GeometryIterator, PartIterator, VecIterator>::element_t>,
  vec_2d<typename multilinestring_range<GeometryIterator, PartIterator, VecIterator>::element_t>>
multilinestring_range<GeometryIterator, PartIterator, VecIterator>::segment(IndexType segment_idx)
{
  return thrust::make_pair(points_begin[segment_idx], points_begin[segment_idx + 1]);
}

}  // namespace cuspatial
