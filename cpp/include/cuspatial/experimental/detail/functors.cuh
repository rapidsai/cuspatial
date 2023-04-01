
#pragma once

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/traits.hpp>

#include <thrust/binary_search.h>

namespace cuspatial {
namespace detail {

/**
 * @brief Given iterator a pair of offsets, return the number of elements between the offsets.
 *
 * Example:
 * pair of offsets: (0, 3), (3, 5), (5, 8)
 * number of elements between offsets: 3, 2, 3
 *
 * @tparam OffsetPairIterator Must be iterator type to thrust::pair of indices.
 * @param p Iterator of thrust::pair of indices.
 */
struct offset_pair_to_count_functor {
  template <typename OffsetPairIterator>
  CUSPATIAL_HOST_DEVICE auto operator()(OffsetPairIterator p)
  {
    return thrust::get<1>(p) - thrust::get<0>(p);
  }
};

/**
 * @brief Convert counts of points to counts of segments.
 *
 * A Multilinestring is composed of a series of Linestrings. Each Linestring is composed of a
 * segments. The number of segments in a multilinestring is the number of points in the
 * multilinestring subtracting the number of linestrings.
 *
 * Caveats: This has a strong assumption that the Multilinestring does not contain empty linestring.
 * While each non-empty linestring in the multilinestring can cause 1 invalid segment, an empty
 * multilinestring not introduce and invalid segments since it does not contain any points.
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
 * @brief Given an iterator of offsets, return an iterator of offsets subtracted by the index.
 *
 * @tparam OffsetIterator Iterator type to the offset
 *
 * Caveats: This has a strong assumption that the Multilinestring does not contain empty linestring.
 * While each non-empty linestring in the multilinestring can cause 1 invalid segment, an empty
 * multilinestring not introduce and invalid segments since it does not contain any points.
 */
template <typename OffsetIterator>
struct to_subtracted_by_index_iterator {
  OffsetIterator begin;

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto operator()(IndexType i)
  {
    return begin[i] - i;
  }
};

/// Deduction guide for to_subtracted_by_index_iterator
template <typename OffsetIterator>
to_subtracted_by_index_iterator(OffsetIterator) -> to_subtracted_by_index_iterator<OffsetIterator>;

/**
 * @brief Return a segment from the a partitioned range of points
 *
 * Used in a counting transform iterator. Given an index of the segment, offset it by the number of
 * skipped segments preceding i in the partitioned range of points. Dereference the corresponding
 * point and the point following to make a segment.
 *
 * @tparam OffsetIterator the iterator type indicating partitions of the point range.
 * @tparam CoordinateIterator the iterator type to the point range.
 */
template <typename OffsetIterator, typename CoordinateIterator>
struct to_valid_segment_functor {
  using element_t = iterator_vec_base_type<CoordinateIterator>;

  OffsetIterator begin;
  OffsetIterator end;
  CoordinateIterator point_begin;

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE segment<element_t> operator()(IndexType i)
  {
    auto kit = thrust::upper_bound(thrust::seq, begin, end, i);
    auto k   = thrust::distance(begin, kit);
    auto pid = i + k - 1;

    return segment<element_t>{point_begin[pid], point_begin[pid + 1]};
  }
};

/// Deduction guide for to_valid_segment_functor
template <typename OffsetIterator, typename CoordinateIterator>
to_valid_segment_functor(OffsetIterator, OffsetIterator, CoordinateIterator)
  -> to_valid_segment_functor<OffsetIterator, CoordinateIterator>;

}  // namespace detail
}  // namespace cuspatial