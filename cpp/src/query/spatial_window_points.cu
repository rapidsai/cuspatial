/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <thrust/count.h>

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/legacy/column.hpp>

#include <utility/utility.hpp>
#include <cuspatial/query.hpp>

namespace {

/**
 *@brief Thrust functor for spatial window query on point data (x/y)
 *
*/
template<typename T>
struct spatial_window_functor_xy
{
    T left, bottom, right, top;

    __device__
    spatial_window_functor_xy(T left, T bottom, T right, T top)
    : left(left), bottom(bottom), right(right), top(top) {}

    __device__
    bool operator()(const thrust::tuple<T, T>& t)
    {
        T x= thrust::get<0>(t);
        T y= thrust::get<1>(t);
        return x > left && x < right && y > bottom && y < top;
    }
};

struct sw_point_functor
{
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<T>::value;
    }

    template <typename T>
    T get_scalar(const gdf_scalar v) {
        T ret{};  // Safe type pun, compiler should optimize away the memcpy
        memcpy(&ret, &v.data, sizeof(T));
        return ret;
    }

    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    std::pair<gdf_column,gdf_column> operator()(const gdf_scalar left,
                                                const gdf_scalar bottom,
                                                const gdf_scalar right,
                                                const gdf_scalar top,
                                                const gdf_column& x,
                                                const gdf_column& y)
    {
        T q_left = get_scalar<T>(left);
        T q_right = get_scalar<T>(right);
        T q_bottom = get_scalar<T>(bottom);
        T q_top = get_scalar<T>(top);

        CUDF_EXPECTS(q_left < q_right,
                     "left must be less than right in a spatial window query");
        CUDF_EXPECTS(q_bottom < q_top,
                     "bottom must be less than top in a spatial window query");

        cudaStream_t stream{0};
        auto exec_policy = rmm::exec_policy(stream)->on(stream);

        auto in_it = thrust::make_zip_iterator(thrust::make_tuple(
            static_cast<T*>(x.data), static_cast<T*>(y.data)));

        int num_hits =
            thrust::count_if(exec_policy, in_it, in_it + x.size,
                             spatial_window_functor_xy<T>(q_left, q_bottom,
                                                          q_right, q_top));

        T* temp_x{nullptr};
        T* temp_y{nullptr};
        RMM_TRY( RMM_ALLOC(&temp_x, num_hits * sizeof(T), 0) );
        RMM_TRY( RMM_ALLOC(&temp_y, num_hits * sizeof(T), 0) );

        auto out_it =
            thrust::make_zip_iterator(thrust::make_tuple(temp_x, temp_y));
        thrust::copy_if(exec_policy, in_it, in_it + x.size, out_it,
                        spatial_window_functor_xy<T>(q_left, q_bottom,
                                                     q_right, q_top));

        gdf_column out_x{}, out_y{};

        gdf_column_view_augmented(&out_x, temp_x, nullptr, num_hits, x.dtype,
                              0, gdf_dtype_extra_info{TIME_UNIT_NONE}, "x");
        gdf_column_view_augmented(&out_y, temp_y, nullptr, num_hits, y.dtype,
                              0, gdf_dtype_extra_info{TIME_UNIT_NONE}, "y");
        return std::make_pair(out_x, out_y);
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr>
    std::pair<gdf_column,gdf_column> operator()(const gdf_scalar left,
                                                const gdf_scalar bottom,
                                                const gdf_scalar right,
                                                const gdf_scalar top,
                                                const gdf_column& x,
                                                const gdf_column& y)
    {
       CUDF_FAIL("Non-floating point operation is not supported");
    }
};

} // namespace anonymous

namespace cuspatial {

/*
 * Return all points (x,y) that fall within a query window (x1,y1,x2,y2)
 * see query.hpp
 */
std::pair<gdf_column,gdf_column> spatial_window_points(const gdf_scalar& left,
                                                       const gdf_scalar& bottom,
                                                       const gdf_scalar& right,
                                                       const gdf_scalar& top,
                                                       const gdf_column& x,
                                                       const gdf_column& y)
{
    CUDF_EXPECTS(x.dtype == y.dtype, "point type mismatch between x/y arrays");
    CUDF_EXPECTS(x.size == y.size, "#of points mismatch between x/y arrays");

    CUDF_EXPECTS(x.null_count == 0 && y.null_count == 0,
                 "this version does not support point data that contains nulls");

    std::pair<gdf_column,gdf_column> res =
        cudf::type_dispatcher(x.dtype, sw_point_functor(), left, bottom,
                              right, top, x, y);

    return res;
}

}// namespace cuspatial
