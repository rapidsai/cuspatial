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
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/gather.h>
#include <limits>
#include <iterator>


template <class OutputIterator>
class haus_output_iterator_proxy;

// namespace thrust {
// namespace detail {
//     template <class OutputIterator>
//     struct is_proxy_reference<haus_output_iterator_proxy<OutputIterator>>
//         : public thrust::detail::true_type {};
// }
// }

namespace {

using size_type = int32_t;

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

template<typename T>
using haus = thrust::tuple<int32_t, int32_t, int32_t, int32_t, T, T, T, int64_t>;

template<typename T> __device__ int32_t haus_col(haus<T> value) { return thrust::get<0>(value); }
template<typename T> __device__ int32_t haus_row(haus<T> value) { return thrust::get<1>(value); }
template<typename T> __device__ int32_t cell_min(haus<T> value) { return thrust::get<2>(value); }
template<typename T> __device__ int32_t cell_max(haus<T> value) { return thrust::get<3>(value); }
template<typename T> __device__ T haus_max(haus<T> value) { return thrust::get<4>(value); }
template<typename T> __device__ T haus_min(haus<T> value) { return thrust::get<5>(value); }
template<typename T> __device__ T haus_res(haus<T> value) { return thrust::get<6>(value); }
template<typename T> __device__ int64_t haus_dst(haus<T> value) { return thrust::get<7>(value); }

template<typename T>
struct haus_key_compare
{
    bool __device__ operator()(haus<T> a, haus<T> b)
    {
        return haus_col(a) == haus_col(b)
            && haus_row(a) == haus_row(b);
    }
};

template<typename T>
struct haus_reduce
{
    haus<T> __device__ operator()(haus<T> lhs, haus<T> rhs)
    {
        T new_min{};
        T new_max{};

        if (cell_max(lhs) == cell_min(rhs))
        {
            new_min = min(haus_min(lhs), haus_min(rhs));
            new_max = haus_max(lhs);
        }
        else
        {
            new_min = haus_min(rhs);
            new_max = max(haus_res(lhs), haus_max(rhs));
        }

        return haus<T>{
            haus_col(lhs),
            haus_row(lhs),
            cell_min(lhs),
            cell_max(rhs),
            new_max,
            new_min,
            max(new_max, new_min),
            haus_dst(rhs)
        };
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

template <typename OutputIterator>
class haus_output_iterator_proxy
{
  public:
    __host__ __device__
    haus_output_iterator_proxy(const OutputIterator& out, const OutputIterator& begin) : out(out), begin(begin)
    {
    }

    __thrust_exec_check_disable__
    template <typename T>
    __host__ __device__
    haus_output_iterator_proxy operator=(const T& x)
    {
        if (haus_dst(x) >= 0) {
            *(begin + haus_dst(x)) = x;
        }

        return *this;
    }

  private:
    OutputIterator out;
    OutputIterator begin;
};

template<typename OutputIterator>
class haus_output_iterator;

template <typename OutputIterator>
struct haus_output_iterator_base
{
    typedef thrust::iterator_adaptor
    <
        haus_output_iterator<OutputIterator>
      , OutputIterator
      , thrust::use_default
      , thrust::use_default
      , thrust::use_default
      , haus_output_iterator_proxy<OutputIterator>
    > type;
};

template<typename OutputIterator>
class haus_output_iterator : public haus_output_iterator_base<OutputIterator>::type
{
public:
    typedef typename
    haus_output_iterator_base<OutputIterator>::type
    super_t;

    friend class thrust::iterator_core_access;

    __host__ __device__
    haus_output_iterator(OutputIterator const& out, OutputIterator const& begin) : super_t(out), begin(begin) {}
    
 private:
    __host__ __device__
    typename super_t::reference dereference() const
    {
        return haus_output_iterator_proxy<OutputIterator>(this->base_reference(), begin);
    }

    OutputIterator begin;
};

template <typename OutputIterator>
haus_output_iterator<OutputIterator>
__host__ __device__
make_haus_output_iterator(OutputIterator out)
{
    return haus_output_iterator<OutputIterator>(out, out);
}

template<typename T, typename SpaceSizeIterator>
struct haus_travesal
{
    int64_t num_spaces;
    int64_t n;
    size_type const* o;
    size_type const* l;
    SpaceSizeIterator const s;
    cudf::column_device_view xs;
    cudf::column_device_view ys;

    haus<T> __device__ operator()(int64_t idx)
    {
        // ===== Reduction Key ===========
        int64_t haus_col = l[idx / n];
        int64_t ox = o[haus_col];
        int64_t sx = s[haus_col];
        int64_t ox_n = ox * n;

        int64_t haus_row = l[(idx - ox_n) / sx];
        int64_t oy = o[haus_row];
        int64_t sy = s[haus_row];

        // ===== Min/Max Key ==========
        int64_t haus_offset = ox_n + sx * oy;
        int64_t cell_idx = idx - haus_offset;
        int64_t cell_col = cell_idx / sy;

        // ===== Distance =============
        int64_t cell_offset = ox_n + oy + (n - sy) * cell_col + cell_idx;
        int64_t col = cell_offset / n;
        int64_t row = cell_offset % n;
        T a_x = xs.element<T>(row);
        T a_y = ys.element<T>(row);
        T b_x = xs.element<T>(col);
        T b_y = ys.element<T>(col);

        double distance_d = hypot(static_cast<double>(b_x - a_x),
                                  static_cast<double>(b_y - a_y));

        T distance = static_cast<T>(distance_d);

        // int64_t distance = abs(b_x - a_x);

        // int64_t elm = ox_n + sx * oy + sx * sy - 1;
        // int64_t dst = haus_col * num_spaces + haus_row;
        int64_t elm = ox_n + (sx - 1) * n + oy + sy - 1;
        // ===== All ==================
        return haus<T>{
            haus_col,
            haus_row,
            cell_col,
            cell_col,
            0,
            distance,
            distance,
            elm == cell_offset ? haus_col * num_spaces + haus_row : -1
        };
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
        size_type num_points = xs.size();
        size_type num_spaces = space_offsets.size();
        int64_t num_results = static_cast<int64_t>(num_spaces) * static_cast<int64_t>(num_spaces);

        if (num_results == 0)
        {
            return make_column<T>(0, stream, mr);
        }

        // ===== Make Space Lookup ================================================================

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

        // ===== Make Space Size Iterator =========================================================

        auto count = thrust::make_counting_iterator<int64_t>(0);

        auto d_space_offsets = cudf::column_device_view::create(space_offsets);

        auto space_offset_iterator = thrust::make_transform_iterator(
            count,
            size_from_offsets_functor { *d_space_offsets, xs.size() }
        );

        // ===== Make Cartesian Distances =========================================================

        auto d_xs = cudf::column_device_view::create(xs);
        auto d_ys = cudf::column_device_view::create(ys);

        auto num_cartesian = static_cast<int64_t>(num_points) * static_cast<int64_t>(num_points);

        auto hausdorff_iter = thrust::make_transform_iterator(
            count,
            haus_travesal<T, decltype(space_offset_iterator)>{
                num_spaces,
                num_points,
                space_offsets.data<size_type>(),
                temp_space_lookup.data().get(),
                space_offset_iterator,
                *d_xs,
                *d_ys
            }
        );

        // ===== Calculate ========================================================================

        std::unique_ptr<cudf::column> result = make_column<T>(num_results, stream, mr);

        auto out_real = result->mutable_view().begin<T>();

        auto discard_buffer = rmm::device_buffer(sizeof(haus<T>) * num_results);
        
        auto discard_pointer_st = static_cast<int32_t*>(discard_buffer.data());
        auto discard_pointer_l = static_cast<int64_t*>(discard_buffer.data());
        auto discard_pointer_t = static_cast<T*>(discard_buffer.data());

        auto out_zip = thrust::make_zip_iterator(
            thrust::make_tuple(
                discard_pointer_st,
                discard_pointer_st,
                discard_pointer_st,
                discard_pointer_st,
                discard_pointer_t,
                discard_pointer_t,
                out_real,
                discard_pointer_l
            )
        );

        auto out = make_haus_output_iterator(out_zip);

        thrust::inclusive_scan_by_key(
            rmm::exec_policy(stream)->on(stream),
            hausdorff_iter,
            hausdorff_iter + num_cartesian,
            hausdorff_iter,
            out,
            haus_key_compare<T>{},
            haus_reduce<T>{}
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
