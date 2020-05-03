// /*
//  * Copyright (c) 2020, NVIDIA CORPORATION.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#include <cmath>
#include <memory>
#include <type_traits>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cuspatial/error.hpp>
#include "rmm/mr/device/device_memory_resource.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include <cuspatial/constants.hpp>

namespace {

template<typename T>
__device__ T calculate_haversine_distance(T radius, T a_lon, T a_lat, T b_lon, T b_lat) {
    auto ax = a_lon * DEGREE_TO_RADIAN;
    auto ay = a_lat * DEGREE_TO_RADIAN;
    auto bx = b_lon * DEGREE_TO_RADIAN;
    auto by = b_lat * DEGREE_TO_RADIAN;

    // haversine formula
    auto x = (bx - ax) / 2;
    auto y = (by - ay) / 2;
    auto sinysqrd = std::sin(y)  * std::sin(y);
    auto sinxsqrd = std::sin(x)  * std::sin(x);
    auto scale    = std::cos(ay) * std::cos(by);

    return 2 * radius * std::asin(std::sqrt(sinysqrd + sinxsqrd * scale));
};

struct haversine_functor
{
    template<typename T, typename... Args>
    std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
    operator()(Args&&...)
    {
        CUSPATIAL_FAIL("haversine_distance does not support non-floating-point types.");
    }

    template<typename T>
    std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>>
    operator()(cudf::column_view const& a_lon,
               cudf::column_view const& a_lat,
               cudf::column_view const& b_lon,
               cudf::column_view const& b_lat,
               double radius,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream)
    {
        auto size = a_lon.size();
        auto tid = cudf::experimental::type_to_id<T>();
        auto result = cudf::make_fixed_width_column(cudf::data_type{ tid },
                                                    size,
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);

        if (size == 0)
        {
            return result;
        }

        auto input_tuple = thrust::make_tuple(thrust::make_constant_iterator<T>(radius),
                                              a_lon.begin<T>(),
                                              a_lat.begin<T>(),
                                              b_lon.begin<T>(),
                                              b_lat.begin<T>());

        auto input_iter = thrust::make_zip_iterator(input_tuple);

        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          input_iter,
                          input_iter + size,
                          result->mutable_view().begin<T>(),
                          [] __device__ (auto inputs) {
                              return calculate_haversine_distance(thrust::get<0>(inputs),
                                                                  thrust::get<1>(inputs),
                                                                  thrust::get<2>(inputs),
                                                                  thrust::get<3>(inputs),
                                                                  thrust::get<4>(inputs));
                          });

        return result;
    }
};

} // anonymous namespace

namespace cuspatial {

namespace detail {

std::unique_ptr<cudf::column> haversine_distance(cudf::column_view const& a_lon,
                                                 cudf::column_view const& a_lat,
                                                 cudf::column_view const& b_lon,
                                                 cudf::column_view const& b_lat,
                                                 double radius,
                                                 rmm::mr::device_memory_resource* mr,
                                                 cudaStream_t stream)
{
    CUSPATIAL_EXPECTS(radius > 0,
                      "radius must be positive.");

    CUSPATIAL_EXPECTS(not a_lon.has_nulls() and
                      not a_lat.has_nulls() and
                      not b_lon.has_nulls() and
                      not b_lat.has_nulls(),
                      "locations must not contain nulls.");

    CUSPATIAL_EXPECTS(a_lat.type() == a_lon.type() and
                      b_lon.type() == a_lon.type() and
                      b_lat.type() == a_lon.type(),
                      "locations must have the same type.");

    CUSPATIAL_EXPECTS(a_lat.size() == a_lon.size() and
                      b_lon.size() == a_lon.size() and
                      b_lat.size() == a_lon.size(),
                      "locations must have the same size.");

    return cudf::experimental::type_dispatcher(a_lon.type(), haversine_functor{}, a_lon, a_lat, b_lon, b_lat, radius, mr, stream);
}

} // namspace detail

std::unique_ptr<cudf::column> haversine_distance(cudf::column_view const& a_lon,
                                                 cudf::column_view const& a_lat,
                                                 cudf::column_view const& b_lon,
                                                 cudf::column_view const& b_lat,
                                                 double radius,
                                                 rmm::mr::device_memory_resource* mr)
{
    return cuspatial::detail::haversine_distance(a_lon, a_lat, b_lon, b_lat, radius, mr, 0);
}

} // namespace cuspatial
