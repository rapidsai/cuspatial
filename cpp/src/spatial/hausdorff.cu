/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iomanip>
#include <memory>
#include <ostream>
#include <type_traits>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cuspatial/error.hpp>
#include <cudf/utilities/error.hpp>
#include "rmm/mr/device/device_memory_resource.hpp"
#include "rmm/thrust_rmm_allocator.h"

#include <thrust/iterator/discard_iterator.h>

namespace {

using size_type = cudf::size_type;
using position = thrust::tuple<size_type, size_type>;

template<typename T>
constexpr auto magnitude_squared(T a, T b) {
    return a * a + b * b;
}

template<typename T>
auto make_column(
    size_type size,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()
)
{
    auto tid = cudf::type_to_id<T>();

    return cudf::make_fixed_width_column(
        cudf::data_type{ tid },
        size,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
}

template<typename T>
struct distance_functor
{
    cudf::column_device_view xs;
    cudf::column_device_view ys;

    T __device__ operator() (size_type idx)
    {
        auto row = idx % xs.size();
        auto col = idx / xs.size();
        auto a_x = xs.element<T>(row);
        auto a_y = ys.element<T>(row);
        auto b_x = xs.element<T>(col);
        auto b_y = ys.element<T>(col);

        return hypot(b_x - a_x, b_y - a_y);
    }
};

struct hausdorff_functor
{
    template<typename T, typename... Args>
    std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
    operator()(Args&&...)
    {
        CUSPATIAL_FAIL("Non-floating point operation is not supported");
    }

    template<typename T>
    std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
    operator()(cudf::column_view const& xs,
               cudf::column_view const& ys,
               cudf::column_view const& space_offsets,
               rmm::mr::device_memory_resource *mr,
               cudaStream_t stream)
    {
        auto num_points = xs.size();
        auto num_spaces = space_offsets.size();
        auto num_results = num_spaces * num_spaces;

        if (num_results == 0)
        {
            return make_column<T>(0, stream, mr);
        }

        // ===== Make Lookup ======================================================================

        auto temp_space_lookup = rmm::device_vector<size_type>(num_points);

        thrust::scatter(
            rmm::exec_policy(stream)->on(stream),
            thrust::make_constant_iterator(1),
            thrust::make_constant_iterator(1) + num_spaces - 1,
            space_offsets.begin<size_type>() + 1,
            temp_space_lookup.begin()
        );

        thrust::inclusive_scan(
            rmm::exec_policy(stream)->on(stream),
            temp_space_lookup.cbegin(),
            temp_space_lookup.cend(),
            temp_space_lookup.begin()
        );

        auto count = thrust::make_counting_iterator<size_type>(0);
        auto lookup_count = thrust::make_transform_iterator(
          count,
          [num_points]
          __device__
          (size_type idx) { return idx % num_points; }
        );
        auto lookup = thrust::make_permutation_iterator(temp_space_lookup.begin(), lookup_count);

        // ===== Make Cartesian Distances =========================================================

        auto d_xs = cudf::column_device_view::create(xs);
        auto d_ys = cudf::column_device_view::create(ys);

        auto num_cartesian = num_points * num_points;
        auto selector = distance_functor<T>{ *d_xs, *d_ys };
        auto cartesian_source = thrust::make_transform_iterator(count, selector);

        // ===== Make Min Reduction ===============================================================

        auto num_minimums = num_spaces * num_points;
        auto temp_minimums = rmm::device_vector<T>(num_minimums);

        thrust::reduce_by_key(
            rmm::exec_policy(stream)->on(stream),
            lookup,
            lookup + num_cartesian,
            cartesian_source,
            thrust::discard_iterator<position>(),
            temp_minimums.begin(),
            thrust::equal_to<position>(),
            thrust::minimum<T>()
        );

        // ===== Make Max Reduction ===============================================================

        auto perm = thrust::make_transform_iterator(
            count,
            [num_spaces, num_points]
            __device__
            (size_type idx) {
              return (idx * num_spaces) % (num_points * num_spaces) +
                     (idx * num_spaces) / (num_points * num_spaces);
            });

        auto temp_minimums_trans = thrust::make_permutation_iterator(temp_minimums.begin(), perm);

        auto result = make_column<T>(num_results, stream, mr);

        thrust::reduce_by_key(
            rmm::exec_policy(stream)->on(stream),
            lookup,
            lookup + num_minimums,
            temp_minimums_trans,
            thrust::discard_iterator<position>(),
            result->mutable_view().template begin<T>(),
            thrust::equal_to<position>(),
            thrust::maximum<T>()
        );

        return result;
    }
};

} // namespace anonymous

namespace cuspatial {

std::unique_ptr<cudf::column>
directed_hausdorff_distance(cudf::column_view const& xs,
                            cudf::column_view const& ys,
                            cudf::column_view const& points_per_space,
                            rmm::mr::device_memory_resource *mr)
{
    CUSPATIAL_EXPECTS(xs.type() == ys.type(), "Inputs `xs` and `ys` must have same type.");
    CUSPATIAL_EXPECTS(xs.size() == ys.size(), "Inputs `xs` and `ys` must have same length.");

    CUSPATIAL_EXPECTS(not xs.has_nulls() and
                      not ys.has_nulls() and
                      not points_per_space.has_nulls(),
                      "Inputs must not have nulls.");

    CUSPATIAL_EXPECTS(xs.size() >= points_per_space.size(),
                      "At least one point is required for each space");

    cudaStream_t stream = 0;

    return cudf::type_dispatcher(xs.type(), hausdorff_functor(),
                                 xs, ys, points_per_space, mr, stream);
}

} // namespace cuspatial
