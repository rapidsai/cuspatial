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
 * @internal
 * @brief Computes the offsets to the starting segment per linestring
 *
 * The point indices and segment indices are correlated, but in a different index space.
 * For example:
 * ```
 * {0, 3}
 * {0, 3, 3, 6}
 * {A, B, C, X, Y, Z}
 * ```
 *
 * ```
 * segments:  AB BC XY YZ
 * sid:       0  1  2  3
 * points:    A B C X Y Z
 * pid:       0 1 2 3 4 5
 * ```
 *
 * The original {0, 3, 3, 6} offsets are in the point index space. For example:
 *  The first and past the last point of the first linestring is at point index 0 and 3 (A, X).
 *  The first and past the last point of the second linestring is at point index 3 and 3 (empty),
 *  and so on.
 *
 * The transformed segment offsets {0, 2, 2, 4} are in the segment index space. For example:
 *  The first and past the last segment of the first linestring is at segment index 0 and 2 ((AB),
 *  (XY)).
 *  The first and past the last segment of the second linestring is at segment index 2 and 2
 *  (empty), and so on.
 *
 * @tparam OffsetIterator Iterator type to the offset
 */
template <typename OffsetIterator, typename CountIterator>
struct point_offset_to_segment_offset {
  OffsetIterator part_offset_begin;
  CountIterator non_empty_linestrings_count_begin;

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto operator()(IndexType i)
  {
    return part_offset_begin[i] - non_empty_linestrings_count_begin[i];
  }
};

/// Deduction guide
template <typename OffsetIterator, typename CountIterator>
point_offset_to_segment_offset(OffsetIterator, CountIterator)
  -> point_offset_to_segment_offset<OffsetIterator, CountIterator>;

/**
 * @internal
 * @brief Given a segment index, return the corresponding segment
 *
 * Given a segment index, first find its corresponding part index by performing a binary search in
 * the segment offsets range. Then, skip the segment index by the number of non empty linestrings
 * that precedes the current linestring to find point index to the first point of the segment.
 * Dereference this point and the following point to construct the segment.
 *
 * @tparam OffsetIterator Iterator to the segment offsets
 * @tparam CountIterator Iterator the the range of the prefix sum of non empty linestrings
 * @tparam CoordinateIterator Iterator to the point range
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
    auto part_id                         = thrust::distance(segment_offset_begin, kit);
    auto preceding_non_empty_linestrings = non_empty_partitions_begin[part_id];
    auto pid                             = sid + preceding_non_empty_linestrings;

    return segment<element_t>{point_begin[pid], point_begin[pid + 1]};
  }
};

/// Deduction guide
template <typename OffsetIterator, typename CountIterator, typename CoordinateIterator>
to_valid_segment_functor(OffsetIterator, OffsetIterator, CountIterator, CoordinateIterator)
  -> to_valid_segment_functor<OffsetIterator, CountIterator, CoordinateIterator>;

/**
 * @internal
 * @brief A non-owning range of segments in a multilinestring
 *
 * A `multilinestring_segment_range` provide views into the segments of a multilinestring.
 * The segments of a multilinestring have a near 1:1 mapping to the points of the multilinestring,
 * except that the last point of a linestring and the first point of the next linestring do not
 * form a valid segment. For example, the below multilinestring (points are denoted a letters):
 *
 * ```
 * {0, 2}
 * {0, 3, 6}
 * {A, B, C, X, Y, Z}
 * ```
 *
 * contains 6 points, but only 4 segments. AB, BC, XY and YZ.
 * If we assign an index to all four segments, and an index to all points:
 *
 * ```
 * segments:  AB BC XY YZ
 * sid:       0  1  2  3
 * points:    A B C X Y Z
 * pid:       0 1 2 3 4 5
 * ```
 *
 * Notice that if we "skip" the segment index by a few steps, it can correctly find the
 * corresponding point index of the starting point of the segment. For example: skipping sid==0 (AB)
 * by 0 steps, finds the starting point of A (pid==0) skipping sid==2 (XY) by 1 step, finds the
 * starting point of X (pid==3)
 *
 * Intuitively, the *steps to skip* equals the number of linestrings that precedes the linestring
 * that the current segment is in. This is because every linestring adds an "invalid" segment to the
 * preceding linestring. However, consider the following edge case that contains empty linestrings:
 *
 * ```
 * {0, 3}
 * {0, 3, 3, 6}
 * {A, B, C, X, Y, Z}
 * ```
 *
 * For segment XY, there are 2 linestrings that precedes its linestring ((0, 3) and (3, 3)).
 * However, we cannot skip the sid of XY by 2 to get its starting point index. This is because the
 * empty linestring in between does not introduce the "invalid" segment. Therefore, the correct
 * steps to skip equals the number of *non-empty* linestrings that precedes the current linestring
 * that the segment is in.
 *
 * Concretely, with the above example:
 * ```
 * segments:                              AB BC XY YZ
 * sid:                                   0  1  2  3
 * num_preceding_non_empty_linestrings:   0  0  1  1
 * skipped sid (pid):                     0  0  3  4
 * starting point:                        A  B  X  Y
 * ```
 *
 * @tparam ParentRange The multilinestring range to construct from
 * @tparam IndexRange The range to the prefix sum of the non empty linestring counts
 */
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

  /// Returns the number of segments in the multilinestring
  CUSPATIAL_HOST_DEVICE index_t num_segments() { return _num_segments; }

  /// Returns starting iterator to the range of the starting segment index per
  /// multilinestring or multipolygon
  CUSPATIAL_HOST_DEVICE auto multigeometry_offset_begin()
  {
    return thrust::make_permutation_iterator(_per_linestring_offset_begin(),
                                             _parent.geometry_offset_begin());
  }

  /// Returns end iterator to the range of the starting segment index per multilinestring
  /// or multipolygon
  CUSPATIAL_HOST_DEVICE auto multigeometry_offset_end()
  {
    return multigeometry_offset_begin() + _parent.num_multilinestrings() + 1;
  }

  /// Returns starting iterator to the range of the number of segments per multilinestring of
  /// multipolygon
  CUSPATIAL_HOST_DEVICE auto multigeometry_count_begin()
  {
    auto zipped_offset_it = thrust::make_zip_iterator(multigeometry_offset_begin(),
                                                      thrust::next(multigeometry_offset_begin()));

    return thrust::make_transform_iterator(zipped_offset_it, offset_pair_to_count_functor{});
  }

  /// Returns end iterator to the range of the number of segments per multilinestring of
  /// multipolygon
  CUSPATIAL_HOST_DEVICE auto multigeometry_count_end()
  {
    return multigeometry_count_begin() + _parent.num_multilinestrings();
  }

  /// Returns the iterator to the first segment of the geometry range
  /// See `to_valid_segment_functor` for implementation detail
  CUSPATIAL_HOST_DEVICE auto begin()
  {
    return make_counting_transform_iterator(
      0,
      to_valid_segment_functor{_per_linestring_offset_begin(),
                               _per_linestring_offset_end(),
                               _non_empty_geometry_prefix_sum.begin(),
                               _parent.point_begin()});
  }

  /// Returns the iterator to the past the last segment of the geometry range
  CUSPATIAL_HOST_DEVICE auto end() { return begin() + _num_segments; }

 private:
  ParentRange _parent;
  IndexRange _non_empty_geometry_prefix_sum;
  index_t _num_segments;

  /// Returns begin iterator to the index that points to the starting index for each linestring
  /// See documentation of `to_segment_offset_iterator` for detail.
  CUSPATIAL_HOST_DEVICE auto _per_linestring_offset_begin()
  {
    return make_counting_transform_iterator(
      0,
      point_offset_to_segment_offset{_parent.part_offset_begin(),
                                     _non_empty_geometry_prefix_sum.begin()});
  }

  /// Returns end iterator to the index that points to the starting index for each linestring
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
