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

#include "segment_method_view.cuh"

#include <cuspatial/range/range.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/detail/functors.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/iterator_factory.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>

namespace cuspatial {
namespace detail {

template<typename IndexType, IndexType value>
struct equals {
    __device__ bool operator()(IndexType x) const {
        return x == value;
    }
};

template<typename IndexType>
struct greater_than_zero_functor {
    __device__ IndexType operator()(IndexType x) const {
        return x > 0;
    }
};

template<typename ParentRange>
class segment_method {

using index_t = iterator_value_type<typename ParentRange::part_it_t>;

public:
    // segment_methods is always internal use, thus memory consumed is always temporary,
    // therefore always use default device memory resource.
    segment_method(ParentRange parent_range, rmm::cuda_stream_view stream) :
        _range(parent_range),
        _non_empty_geometry_prefix_sum(0, stream) {

        auto offset_range = range(_range.part_offset_begin(), _range.part_offset_end());
        auto count_begin = thrust::make_transform_iterator(
            thrust::make_zip_iterator(offset_range.begin(), thrust::next(offset_range.begin())),
            offset_pair_to_count_functor{});

        // // Preemptive test: does the given range contain any empty ring/linestring?
        // _contains_empty_geom = thrust::any_of(
        //     rmm::exec_policy(stream),
        //     count_begin,
        //     count_begin + _range.num_linestrings(),
        //     equals<index_t, 0>{}
        // );

        // std::cout << std::boolalpha << "contains empty geometry: " << _contains_empty_geom << std::endl;

            auto count_greater_than_zero = thrust::make_transform_iterator(
                count_begin,
                greater_than_zero_functor<index_t>{}
            );

            thrust::device_vector<index_t> count_greater_than_zero_truth(
                count_greater_than_zero,
                count_greater_than_zero + _range.num_linestrings());

            test::print_device_vector(
                count_greater_than_zero_truth,
                "count_greater_than_zero_truth: "
            );

            // Compute the number of empty linestrings
            _non_empty_geometry_prefix_sum.resize(
                _range.num_multilinestrings() + 1, stream
            );

            auto key_begin = make_geometry_id_iterator<index_t>(
                _range.geometry_offsets_begin(),
                _range.geometry_offsets_end()
            );

            thrust::device_vector<index_t> key_truth(
                key_begin,
                key_begin + _range.num_linestrings());

            test::print_device_vector(
                key_truth,
                "key_truth: "
            );

            zero_data_async(
                _non_empty_geometry_prefix_sum.begin(),
                _non_empty_geometry_prefix_sum.end(),
                stream
            );
            thrust::reduce_by_key(
                rmm::exec_policy(stream),
                key_begin,
                key_begin + _range.num_linestrings(),
                count_greater_than_zero,
                thrust::make_discard_iterator(),
                thrust::next(_non_empty_geometry_prefix_sum.begin()),
                thrust::equal_to<index_t>{},
                thrust::plus<index_t>{}
            );

            test::print_device_vector(
                _non_empty_geometry_prefix_sum,
                "non_empty_linestrings: ");

            thrust::inclusive_scan(
                rmm::exec_policy(stream),
                thrust::next(_non_empty_geometry_prefix_sum.begin()),
                _non_empty_geometry_prefix_sum.end(),
                thrust::next(_non_empty_geometry_prefix_sum.begin())
            );

            _num_segments = _range.num_points() - _non_empty_geometry_prefix_sum.element(
                _non_empty_geometry_prefix_sum.size() - 1, stream
            );

        test::print_device_vector(
            _non_empty_geometry_prefix_sum,
            "non_empty_geometry_prefix_sum: ");
    }

    auto view() {
        auto index_range = range(
            _non_empty_geometry_prefix_sum.begin(),
            _non_empty_geometry_prefix_sum.end()
        );
        return segment_method_view<ParentRange, decltype(index_range)>{
            _range,
            index_range,
            _num_segments,
            _contains_empty_geom
        };
    }

private:
    ParentRange _range;
    bool _contains_empty_geom;
    index_t _num_segments;
    rmm::device_uvector<index_t> _non_empty_geometry_prefix_sum;
};

}  // namespace detail



}  // namespace cuspatial
