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
