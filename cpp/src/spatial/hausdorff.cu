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
#include <rmm/device_buffer.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/gather.h>
#include <limits>
#include <iterator>

namespace {

using size_type = cudf::size_type;
using position = thrust::tuple<size_type, size_type>;

template<typename T>
constexpr auto magnitude_squared(T a, T b) {
    return a * a + b * b;
}

template<typename T>
std::unique_ptr<cudf::column> make_column(
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

template<typename T> using haus = thrust::tuple<int32_t, int32_t, int32_t, T, T, T>;
template<typename T> __device__ T haus_col(haus<T> value) { return thrust::get<0>(value); }
template<typename T> __device__ T haus_row(haus<T> value) { return thrust::get<1>(value); }
template<typename T> __device__ T cell_col(haus<T> value) { return thrust::get<2>(value); }
template<typename T> __device__ T haus_max(haus<T> value) { return thrust::get<3>(value); }
template<typename T> __device__ T haus_min(haus<T> value) { return thrust::get<4>(value); }
template<typename T> __device__ T haus_res(haus<T> value) { return thrust::get<5>(value); }

template<typename T>
struct haus_key_compare
{
    bool __device__ operator()(haus<T> a, haus<T> b)
    {
        return thrust::get<0>(a) == thrust::get<0>(b) 
            && thrust::get<1>(a) == thrust::get<1>(b);
    }
};

template<typename T>
struct haus_reduce
{
    haus<T> __device__ operator()(haus<T> lhs, haus<T> rhs)
    {
        if (cell_col(lhs) == cell_col(rhs))
        {
            auto new_min = std::min(haus_min(lhs), haus_min(rhs));
            return haus<T>(
                haus_col(lhs),
                haus_row(lhs),
                cell_col(rhs),
                haus_max(lhs),
                new_min,
                std::max(haus_max(lhs), new_min)
            );
        }
        else
        {
            auto new_max = std::max(haus_max(lhs), haus_res(lhs));
            return haus<T>(
                haus_col(lhs),
                haus_row(lhs),
                cell_col(rhs),
                new_max,
                haus_min(rhs),
                std::max(new_max, haus_min(rhs)) // could haus_min just as well be haus_res ?
            );
        }
    }
};

struct size_from_offsets_functor
{
    cudf::column_device_view offsets;
    size_type end;

    size_type __device__ operator()(size_type idx)
    {
        auto curr_offset = offsets.element<size_type>(idx);
        auto next_idx = idx + 1;
        auto next_offset = next_idx >= offsets.size()
            ? end
            : offsets.element<size_type>(next_idx);
        
        return next_offset - curr_offset;
    }
};

template<typename T>
struct haus_travesal
{
    size_type n;
    size_type const* o;
    size_type const* l;
    size_type const* s;
    cudf::column_device_view xs;
    cudf::column_device_view ys;

    haus<T> __device__ operator()(size_type idx)
    {
        // ===== Reduction Key ===========
        auto haus_col = l[idx / n];
        auto o_x = o[haus_col];
        auto s_x = s[haus_col];

        auto haus_row = l[(idx - o_x * n) / s_x];
        auto o_y = o[haus_row];
        auto s_y = s[haus_row];

        // ===== Min/Max Key ==========
        auto haus_offset = n * o_x + s_x * o_y;
        auto cell_idx = idx - haus_offset;
        auto cell_col = cell_idx / s_y;

        // ===== Distance =============
        auto cell_offset = n * o_x + o_y + (n - s_y) * cell_col + cell_idx;
        auto col = cell_offset / n;
        auto row = cell_offset % n;
        auto a_x = xs.element<T>(row);
        auto a_y = ys.element<T>(row);
        auto b_x = xs.element<T>(col);
        auto b_y = ys.element<T>(col);

        // auto distance = magnitude_squared(b_x - a_x, b_y - a_y);
        auto distance = abs(b_x - a_x);

        // ===== All ==================
        return haus<T>{
            haus_col,
            haus_row,
            cell_col,
            0,
            distance,
            distance
            // cell_col,
            // cell_col,
            // cell_col
        };
    }
};

// template<typename Iterator>
// class repeat_iterator : public thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator>
// {
// public:
//     typedef thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator> super_t;

//     __host__ __device__
//     repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}

//     friend class thrust::iterator_core_access;

//     std::iterator_traits<Iterator>::value_type x;

//  private:

//     unsigned int n;

//     const Iterator begin;
//     // it is private because only thrust::iterator_core_access needs access to it
//     __host__ __device__
//     typename super_t::reference dereference() const
//     {
//       return *(begin + (this->base() - begin) / n);
//     }
// };

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
        int32_t num_points = xs.size();
        int64_t num_spaces = space_offsets.size();
        auto num_results = num_points * num_points; // should be num_spaces ^ 2

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

        // ===== Make Lengths =====================================================================

        auto temp_space_size = rmm::device_vector<size_type>(num_spaces);
        
        auto count = thrust::make_counting_iterator<size_type>(0);

        auto d_space_offsets = cudf::column_device_view::create(space_offsets);

        thrust::transform(
            rmm::exec_policy(stream)->on(stream),
            count,
            count + space_offsets.size() + 1,
            temp_space_size.data().get(),
            size_from_offsets_functor { *d_space_offsets, xs.size() }
        );

        // ===== Make Cartesian Distances =========================================================

        auto d_xs = cudf::column_device_view::create(xs);
        auto d_ys = cudf::column_device_view::create(ys);

        int64_t num_cartesian = (int64_t) num_points * num_points;


        auto hausdorff_iter = thrust::make_transform_iterator(
            count,
            haus_travesal<T>{
                num_points,
                space_offsets.data<size_type>(),
                temp_space_lookup.data().get(),
                temp_space_size.data().get(),
                *d_xs,
                *d_ys
            }
        );

        // ===== Calculate ========================================================================

        std::unique_ptr<cudf::column> result = make_column<T>(num_results, stream, mr);

        auto discard_buffer = rmm::device_buffer(sizeof(haus<T>) * num_results);
        auto temp_buffer = rmm::device_buffer(sizeof(T) * num_cartesian);

        auto discard_pointer_st = static_cast<int32_t*>(discard_buffer.data());
        auto discard_pointer_t = static_cast<T*>(discard_buffer.data());

        auto out = thrust::make_zip_iterator(
            thrust::make_tuple(
                discard_pointer_st,
                discard_pointer_st,
                discard_pointer_st,
                discard_pointer_t,
                discard_pointer_t,
                static_cast<T*>(temp_buffer.data())
            )
        );

        thrust::inclusive_scan_by_key(
            rmm::exec_policy(stream)->on(stream),
            hausdorff_iter,
            hausdorff_iter + num_cartesian,
            hausdorff_iter,
            out,
            haus_key_compare<T>{},
            haus_reduce<T>{}
        );

        thrust::copy(
            rmm::exec_policy(stream)->on(stream),
            static_cast<T*>(temp_buffer.data()),
            static_cast<T*>(temp_buffer.data()) + num_results,
            result->mutable_view().begin<T>()
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
