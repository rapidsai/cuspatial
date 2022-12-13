#pragma once

#include "cuspatial/cuda_utils.hpp"
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/geometry/linestring_ref.cuh>

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
  return _point_begin;
}

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multilinestring_ref<PartIterator, VecIterator>::point_end() const
{
  return _point_end;
}

template <typename PartIterator, typename VecIterator>
template <typename IndexType>
CUSPATIAL_HOST_DEVICE auto multilinestring_ref<PartIterator, VecIterator>::operator[](
  IndexType i) const
{
  return *(part_begin() + i);
}

}  // namespace cuspatial
