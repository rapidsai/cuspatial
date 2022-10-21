#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/geometry/linestring_ref.cuh>

#include <thrust/iterator/transform_iterator.h>

namespace cuspatial {

template <typename PartIterator, typename VecIterator>
struct to_linestring_functor {
  PartIterator part_begin;
  VecIterator point_begin;

  template <typename IndexType>
  auto operator()(IndexType i)
  {
    return linestring_ref{point_begin + part_begin[i], point_begin + part_begin[i + 1]};
  }
};

template <typename PartIterator, typename VecIterator>
class multilinestring_ref;

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE multilinestring_ref<PartIterator, VecIterator>::multipoint_ref(
  PartIterator part_begin, PartIterator part_end, VecIterator point_begin, VecIterator point_end)
  : _part_begin(part_begin), _part_end(part_end), _points_begin(point_begin), _points_end(point_end)
{
}

template <typename PartIterator, typename VecIterator>
CUSPATIAL_HOST_DEVICE multilinestring_ref<PartIterator, VecIterator>::num_linestrings()
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
  return part_begin() + size();
}
template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_ref<VecIterator>::point_begin() const
{
  return _points_begin;
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint_ref<VecIterator>::point_end() const
{
  return _points_end;
}

}  // namespace cuspatial
