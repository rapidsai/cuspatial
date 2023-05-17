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

#include <cuspatial_test/test_util.cuh>

#include <cuspatial/detail/functors.cuh>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

namespace cuspatial {
namespace detail {

template <typename ParentRange, typename IndexRange>
class segment_method_view {
  using index_t = typename IndexRange::value_type;

 public:
  segment_method_view(ParentRange range,
                      IndexRange non_empty_geometry_prefix_sum,
                      index_t num_segments,
                      bool contains_empty_ring)
    : _range(range),
      _non_empty_geometry_prefix_sum(non_empty_geometry_prefix_sum),
      _num_segments(num_segments),
      _contains_empty_ring(contains_empty_ring)
  {
  }

  auto non_empty_linestring_count_begin()
  {
    auto begin        = _non_empty_geometry_prefix_sum.begin();
    auto paired_begin = thrust::make_zip_iterator(begin, thrust::next(begin));
    return thrust::make_transform_iterator(paired_begin, offset_pair_to_count_functor{});
  }

  auto segment_count_begin()
  {
    auto num_points_begin = _range.multilinestring_point_count_begin();
    auto n_point_linestring_pair_it =
      thrust::make_zip_iterator(num_points_begin, this->non_empty_linestring_count_begin());

    return thrust::make_transform_iterator(n_point_linestring_pair_it,
                                           point_count_to_segment_count_functor{});
  }

  auto segment_count_end() {
    std::cout << "num multilinestrings: " << _range.num_multilinestrings() << std::endl;
    return thrust::next(this->segment_count_begin(), _range.num_multilinestrings()); }

  index_t num_segments() { return _num_segments; }

  auto segment_offset_begin()
  {
    return make_counting_transform_iterator(
      0,
      to_segment_offset_iterator{_range.part_offset_begin(),
                                 _non_empty_geometry_prefix_sum.begin()});
  }

  auto segment_offset_end() { return segment_offset_begin() + num_segments(); }

  auto segment_begin()
  {
    return make_counting_transform_iterator(
      0,
      to_valid_segment_functor{segment_offset_begin(),
                               segment_offset_end(),
                               _non_empty_geometry_prefix_sum.begin(),
                               _range.point_begin()});
  }

  auto segment_end() { return segment_begin() + _num_segments; }

 private:
  ParentRange _range;
  IndexRange _non_empty_geometry_prefix_sum;
  index_t _num_segments;
  bool _contains_empty_ring;
};

template <typename ParentRange, typename IndexRange>
segment_method_view(ParentRange, IndexRange, typename IndexRange::value_type, bool)
  -> segment_method_view<ParentRange, IndexRange>;

}  // namespace detail

}  // namespace cuspatial
