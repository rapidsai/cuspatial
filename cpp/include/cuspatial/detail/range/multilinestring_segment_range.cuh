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
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cuspatial {
namespace detail {

/**
 * @brief Convert counts of points to counts of segments in a linestring.
 *
 * A Multilinestring is composed of a series of Linestrings. Each Linestring is composed of a series
 * of segments. The number of segments in a multilinestring is the number of points in the
 * multilinestring minus the number of linestrings.
 *
 * Caveats: This has a strong assumption that the Multilinestring does not contain empty
 * linestrings. While each non-empty linestring in the multilinestring represents 1 extra segment,
 * an empty multilinestring does not introduce any extra segments since it does not contain any
 * points.
 *
 * Used to create segment count iterators, such as `multi*_segment_count_begin`.
 *
 * @tparam IndexPair Must be iterator to a pair of counts
 * @param n_point_linestring_pair A pair of counts, the first is the number of points, the second is
 * the number of linestrings.
 */
struct point_count_to_segment_count_functor {
  template <typename IndexPair>
  CUSPATIAL_HOST_DEVICE auto operator()(IndexPair n_point_linestring_pair)
  {
    auto nPoints      = thrust::get<0>(n_point_linestring_pair);
    auto nLinestrings = thrust::get<1>(n_point_linestring_pair);
    return nPoints - nLinestrings;
  }
};

/**
 * @brief Given an offset iterator it that partitions a point range, return an offset iterator that
 * partitions the segment range made from the same point range.
 *
 * One partition to a point range introduces one invalid segment, except empty partitions.
 * Therefore, the offsets that partitions the segment range is the offset that partitions the point
 * range subtracts the number of *non-empty* point partitions that precedes the current point range.
 *
 * @tparam OffsetIterator Iterator type to the offset
 *
 * Caveats: This has a strong assumption that the Multilinestring does not contain empty
 * linestrings. While each non-empty linestring in the multilinestring represents 1 extra segment,
 * an empty multilinestring does not introduce any extra segments since it does not contain any
 * points.
 *
 * Used to create iterator to segment offsets, such as `segment_offset_begin`.
 */
template <typename OffsetIterator, typename CountIterator>
struct to_segment_offset_iterator {
  OffsetIterator point_partition_begin;
  CountIterator non_empty_partitions_begin;

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto operator()(IndexType i)
  {
    return point_partition_begin[i] - non_empty_partitions_begin[i];
  }
};

/// Deduction guide for to_distance_iterator
template <typename OffsetIterator, typename CountIterator>
to_segment_offset_iterator(OffsetIterator, CountIterator)
  -> to_segment_offset_iterator<OffsetIterator, CountIterator>;

/**
 * @brief Return a segment from the a partitioned range of points
 *
 * Used in a counting transform iterator. Given an index of the segment, offset it by the number of
 * skipped segments preceding i in the partitioned range of points. Dereference the corresponding
 * point and the point following to make a segment.
 *
 * Used to create iterator to segments, such as `segment_begin`.
 *
 * @tparam OffsetIterator the iterator type indicating partitions of the point range.
 * @tparam CoordinateIterator the iterator type to the point range.
 */
template <typename OffsetIterator, typename CountIterator, typename CoordinateIterator>
struct to_valid_segment_functor {
  using element_t = iterator_vec_base_type<CoordinateIterator>;

  OffsetIterator segment_offset_begin;
  OffsetIterator segment_offset_end;
  CountIterator non_empty_partitions_begin;
  CoordinateIterator point_begin;

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE segment<element_t> operator()(IndexType sid)
  {
    auto kit =
      thrust::prev(thrust::upper_bound(thrust::seq, segment_offset_begin, segment_offset_end, sid));
    auto geometry_id                     = thrust::distance(segment_offset_begin, kit);
    auto preceding_non_empty_linestrings = non_empty_partitions_begin[geometry_id];
    auto pid                             = sid + preceding_non_empty_linestrings;

    return segment<element_t>{point_begin[pid], point_begin[pid + 1]};
  }
};

/// Deduction guide for to_valid_segment_functor
template <typename OffsetIterator, typename CountIterator, typename CoordinateIterator>
to_valid_segment_functor(OffsetIterator, OffsetIterator, CountIterator, CoordinateIterator)
  -> to_valid_segment_functor<OffsetIterator, CountIterator, CoordinateIterator>;

template <typename ParentRange, typename IndexRange>
class multilinestring_segment_range {
  using index_t = typename IndexRange::value_type;

 public:
  multilinestring_segment_range(ParentRange parent,
                                IndexRange non_empty_geometry_prefix_sum,
                                index_t num_segments)
    : _parent(parent),
      _non_empty_geometry_prefix_sum(non_empty_geometry_prefix_sum),
      _num_segments(num_segments)
  {
  }

  CUSPATIAL_HOST_DEVICE index_t num_segments() { return _num_segments; }

  CUSPATIAL_HOST_DEVICE auto multigeometry_offset_begin()
  {
    return thrust::make_permutation_iterator(_per_linestring_offset_begin(),
                                             _parent.geometry_offsets_begin());
  }

  CUSPATIAL_HOST_DEVICE auto multigeometry_offset_end()
  {
    return multigeometry_offset_begin() + _parent.num_multilinestrings() + 1;
  }

  CUSPATIAL_HOST_DEVICE auto multigeometry_count_begin()
  {
    auto zipped_offset_it = thrust::make_zip_iterator(multigeometry_offset_begin(),
                                                      thrust::next(multigeometry_offset_begin()));

    return thrust::make_transform_iterator(zipped_offset_it, offset_pair_to_count_functor{});
  }

  CUSPATIAL_HOST_DEVICE auto multigeometry_count_end()
  {
    return multigeometry_count_begin() + _parent.num_multilinestrings();
  }

  CUSPATIAL_HOST_DEVICE auto begin()
  {
    return make_counting_transform_iterator(
      0,
      to_valid_segment_functor{_per_linestring_offset_begin(),
                               _per_linestring_offset_end(),
                               _non_empty_geometry_prefix_sum.begin(),
                               _parent.point_begin()});
  }

  CUSPATIAL_HOST_DEVICE auto end() { return begin() + _num_segments; }

 private:
  ParentRange _parent;
  IndexRange _non_empty_geometry_prefix_sum;
  index_t _num_segments;

  CUSPATIAL_HOST_DEVICE auto _per_linestring_offset_begin()
  {
    return make_counting_transform_iterator(
      0,
      to_segment_offset_iterator{_parent.part_offset_begin(),
                                 _non_empty_geometry_prefix_sum.begin()});
  }

  CUSPATIAL_HOST_DEVICE auto _per_linestring_offset_end()
  {
    return _per_linestring_offset_begin() + _non_empty_geometry_prefix_sum.size();
  }
};

template <typename ParentRange, typename IndexRange>
multilinestring_segment_range(ParentRange, IndexRange, typename IndexRange::value_type, bool)
  -> multilinestring_segment_range<ParentRange, IndexRange>;

}  // namespace detail

}  // namespace cuspatial
