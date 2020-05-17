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

template<typename Iterator>
void print_table(Iterator source, size_type rows, size_type columns)
{
    for (size_type row = 0; row < rows; row++) {
        for (size_type col = 0; col < columns; col++) {
            auto idx = (col * rows) + row;
            std::cout << *(source + idx) << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
auto make_column(
    size_type size,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()
)
{
    auto tid = cudf::experimental::type_to_id<T>();

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
        return (row + 1) * 10 + (col + 1);
    }
};

template<typename T, typename SpaceLookup>
struct min_key_functor
{
    size_type num_points;
    SpaceLookup space_lookup;

    position __device__ operator() (size_type idx)
    {
        size_type row = idx % num_points;
        size_type col = idx / num_points;
        return thrust::make_tuple(*(space_lookup + row), col);
    }
};

template<typename T, typename SpaceLookup>
struct max_key_functor
{
    size_type num_spaces;
    SpaceLookup space_lookup;

    position __device__ operator() (size_type idx)
    {
        auto row = idx % num_spaces;
        auto col = idx / num_spaces;
        return thrust::make_tuple(row, *(space_lookup + col));
    }
};

template<typename T>
struct repr_key
{
    T __device__ operator()(position idx){

        auto row = thrust::get<0>(idx);
        auto col = thrust::get<1>(idx);
        return (row + 1) * 10 + (col + 1);

        // return thrust::get<1>(idx) * 10 11 + thrust::get<0>(idx);
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

        // ===== Make Device ======================================================================

        auto d_xs = cudf::column_device_view::create(xs);
        auto d_ys = cudf::column_device_view::create(xs);
        auto d_space_offsets = cudf::column_device_view::create(xs);

        // ===== Make Lookup ======================================================================

        std::cout << "===== Lookup =================================================" << std::endl;

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

        rmm::device_vector<size_type> h_temp_space_lookup = temp_space_lookup;

        print_table(h_temp_space_lookup.begin(), 1, num_points);

        // ===== Make Cartesian ===================================================================

        std::cout << "===== Cartesian ==============================================" << std::endl;

        auto num_cartesian = num_points * num_points;
        auto temp_cartesian = rmm::device_vector<T>(num_cartesian);

        auto selector = distance_functor<T>{ *d_xs, *d_ys };

        thrust::transform(
            rmm::exec_policy(stream)->on(stream),
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + num_cartesian,
            temp_cartesian.begin(),
            selector
        );

        thrust::host_vector<T> h_temp_cartesian = temp_cartesian;

        print_table(h_temp_cartesian.begin(), num_points, num_points);

        // ===== Make Min Reduction ===============================================================

        std::cout << "===== Min Reduction ==========================================" << std::endl;

        auto num_minimums = num_spaces * num_points;
        auto temp_minimums = rmm::device_vector<T>(num_minimums);

        auto count = thrust::make_counting_iterator<size_type>(0);

        auto min_key_iter = thrust::make_transform_iterator(
            count,
            min_key_functor<T, decltype(temp_space_lookup.begin())>{num_points, temp_space_lookup.begin()}
        );

        thrust::reduce_by_key(
            rmm::exec_policy(stream)->on(stream),
            min_key_iter,
            min_key_iter + num_cartesian,
            temp_cartesian.begin(),
            thrust::discard_iterator<position>(),
            temp_minimums.begin(),
            thrust::equal_to<position>(),
            thrust::minimum<T>()
        );

        thrust::host_vector<T> h_temp_minimums = temp_minimums;
        
        // print_table(h_temp_minimums.begin(), num_spaces, num_points);

        // std::cout << "===== Min Reduction : Boop ===================================" << std::endl;

        auto min_key_repr_iter = thrust::make_transform_iterator(min_key_iter, repr_key<size_type>{});

        thrust::copy(
            rmm::exec_policy(stream)->on(stream),
            min_key_repr_iter,
            min_key_repr_iter + num_cartesian,
            temp_cartesian.begin()
        );

        h_temp_cartesian = temp_cartesian;
        print_table(h_temp_cartesian.begin(), num_points, num_points);
        std::cout << "===== Min Reduction : Boop ===================================" << std::endl;
        print_table(h_temp_minimums.begin(), num_spaces, num_points);

        // ===== Make Max Reduction ===============================================================

        std::cout << "===== Max Reduction ==========================================" << std::endl;

        rmm::device_vector<T> temp_maximums(num_results);

        auto max_key_iter = thrust::make_transform_iterator(
            count,
            max_key_functor<T, decltype(temp_space_lookup.begin())>{num_spaces, temp_space_lookup.begin()}
        );

        thrust::reduce_by_key(
            rmm::exec_policy(stream)->on(stream),
            max_key_iter,
            max_key_iter + num_minimums,
            temp_minimums.begin(),
            thrust::discard_iterator<position>(),
            temp_maximums.begin(),
            thrust::equal_to<position>(),
            thrust::plus<T>()
        );

        thrust::host_vector<T> h_temp_maximums = temp_maximums;

        // print_table(h_temp_maximums.begin(), num_spaces, num_spaces);

        // std::cout << "===== Max Reduction : Boop ===================================" << std::endl;

        auto max_key_repr_iter = thrust::make_transform_iterator(max_key_iter, repr_key<size_type>{});

        thrust::copy(
            rmm::exec_policy(stream)->on(stream),
            max_key_repr_iter,
            max_key_repr_iter + num_minimums,
            temp_minimums.begin()
        );

        h_temp_minimums = temp_minimums;
        print_table(temp_minimums.begin(), num_spaces, num_points);
        std::cout << "===== Max Reduction : Boop ===================================" << std::endl;
        print_table(h_temp_maximums.begin(), num_spaces, num_spaces);

        std::cout << "===== Return =================================================" << std::endl;

        return make_column<T>(1, stream, mr);
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

    return cudf::experimental::type_dispatcher(xs.type(), hausdorff_functor(),
                                               xs, ys, points_per_space, mr, stream);
}

} // namespace cuspatial
