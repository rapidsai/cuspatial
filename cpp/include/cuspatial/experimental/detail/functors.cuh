
#pragma once

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/traits.hpp>

#include <thrust/binary_search.h>

namespace cuspatial {
namespace detail {

struct offset_pair_to_count_functor {
  template <typename OffsetPairIterator>
  CUSPATIAL_HOST_DEVICE auto operator()(OffsetPairIterator p)
  {
    auto first  = thrust::get<0>(p);
    auto second = thrust::get<1>(p);
    return second - first;
  }
};

struct point_count_to_segment_count_functor {
  template <typename IndexPair>
  CUSPATIAL_HOST_DEVICE auto operator()(IndexPair n_point_linestring_pair)
  {
    auto nPoints      = thrust::get<0>(n_point_linestring_pair);
    auto nLinestrings = thrust::get<1>(n_point_linestring_pair);
    return nPoints - nLinestrings;
  }
};

template <typename OffsetIterator>
struct to_subtracted_by_index_iterator {
  OffsetIterator begin;

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto operator()(IndexType i)
  {
    // printf("begin[i]: %d i: %d\n", static_cast<int>(begin[i]), static_cast<int>(i));
    return begin[i] - i;
  }
};

template <typename OffsetIterator>
to_subtracted_by_index_iterator(OffsetIterator) -> to_subtracted_by_index_iterator<OffsetIterator>;

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

    // printf("%d %d %d\n", static_cast<int>(i), static_cast<int>(k), static_cast<int>(pid));
    return segment<element_t>{point_begin[pid], point_begin[pid + 1]};
  }
};

template <typename OffsetIterator, typename CoordinateIterator>
to_valid_segment_functor(OffsetIterator, OffsetIterator, CoordinateIterator)
  -> to_valid_segment_functor<OffsetIterator, CoordinateIterator>;

template <typename IndexType>
struct wraparound_functor {
  IndexType length;

  template <typename IndexType2>
  CUSPATIAL_HOST_DEVICE auto operator()(IndexType2 i)
  {
    return i % length;
  }
};

template <typename IndexType>
wraparound_functor(IndexType) -> wraparound_functor<IndexType>;

// template <typename IndexType>
// struct repeat_functor {
//   IndexType repeats;

//   template <typename IndexType2>
//   CUSPATIAL_HOST_DEVICE auto operator()(IndexType2 i)
//   {
//     return i / repeats;
//   }
// };

// template <typename IndexType>
// wraparound_functor(IndexType) -> wraparound_functor<IndexType>;

}  // namespace detail
}  // namespace cuspatial
