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

#include <cuspatial/detail/functors.cuh>
#include <cuspatial/detail/range/multilinestring_segment_range.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/range.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

namespace cuspatial {
namespace detail {

template <typename IndexType>
struct greater_than_zero_functor {
  __device__ IndexType operator()(IndexType x) const { return x > 0; }
};

// Optimization: for range that does not contain any empty linestrings,
// The _non_empty_linestring_prefix_sum can be initailized with counting_iterator.
template <typename MultilinestringRange>
class multilinestring_segment {
  using index_t = iterator_value_type<typename MultilinestringRange::part_it_t>;

 public:
  // segment_methods is always internal use, thus memory consumed is always temporary,
  // therefore always use default device memory resource.
  multilinestring_segment(MultilinestringRange parent, rmm::cuda_stream_view stream)
    : _parent(parent), _non_empty_linestring_prefix_sum(parent.num_linestrings() + 1, stream)
  {
    auto offset_range = ::cuspatial::range{_parent.part_offset_begin(), _parent.part_offset_end()};
    auto count_begin  = thrust::make_transform_iterator(
      thrust::make_zip_iterator(offset_range.begin(), thrust::next(offset_range.begin())),
      offset_pair_to_count_functor{});

    auto count_greater_than_zero =
      thrust::make_transform_iterator(count_begin, greater_than_zero_functor<index_t>{});

    zero_data_async(
      _non_empty_linestring_prefix_sum.begin(), _non_empty_linestring_prefix_sum.end(), stream);

    thrust::inclusive_scan(rmm::exec_policy(stream),
                           count_greater_than_zero,
                           count_greater_than_zero + _parent.num_linestrings(),
                           thrust::next(_non_empty_linestring_prefix_sum.begin()));

    _num_segments = _parent.num_points() - _non_empty_linestring_prefix_sum.element(
                                             _non_empty_linestring_prefix_sum.size() - 1, stream);
  }

  auto view()
  {
    auto index_range = ::cuspatial::range{_non_empty_linestring_prefix_sum.begin(),
                                          _non_empty_linestring_prefix_sum.end()};
    return multilinestring_segment_range<MultilinestringRange, decltype(index_range)>{
      _parent, index_range, _num_segments};
  }

 private:
  MultilinestringRange _parent;
  index_t _num_segments;
  rmm::device_uvector<index_t> _non_empty_linestring_prefix_sum;
};

}  // namespace detail

}  // namespace cuspatial
