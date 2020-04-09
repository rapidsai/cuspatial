/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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


#include <type_traits>
#include <utility>
#include <thrust/for_each.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cuspatial/error.hpp>

namespace {

using pair_of_columns = std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>;

struct ll2coord_functor
{
    template <typename T, typename... Args>
    std::enable_if_t<not std::is_floating_point<T>::value, pair_of_columns>
    operator()(Args&&...)
    {
        CUSPATIAL_FAIL("Non-floating point operation is not supported");
    }

    template <typename T>
    std::enable_if_t<std::is_floating_point<T>::value, pair_of_columns>
    operator()(double origin_lon,
               double origin_lat,
               cudf::column_view const& input_lon,
               cudf::column_view const& input_lat,
               rmm::mr::device_memory_resource *mr,
               cudaStream_t stream)
    {
        auto size = input_lon.size();
        auto tid = cudf::experimental::type_to_id<T>();
        auto type = cudf::data_type{ tid };
        auto d_input_lon = cudf::column_device_view::create(input_lon);
        auto d_input_lat = cudf::column_device_view::create(input_lat);
        auto output_lon = cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);
        auto output_lat = cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);
        auto d_output_lon = cudf::mutable_column_device_view::create(output_lon->mutable_view());
        auto d_output_lat = cudf::mutable_column_device_view::create(output_lat->mutable_view());

        auto iter = thrust::make_counting_iterator<cudf::size_type>(0);

        thrust::for_each(iter,
                         iter + size,
                         [origin_lon=origin_lon,    origin_lat=origin_lat,
                              in_lon=*d_input_lon,      in_lat=*d_input_lat,
                             out_lon=*d_output_lon,    out_lat=*d_output_lat]
                         __device__(cudf::size_type idx) mutable
                         {
                            out_lon.element<T>(idx) = (origin_lon - in_lon.element<T>(idx))
                                * 40000.0
                                * cos((origin_lat + in_lon.element<T>(idx)) * M_PI / 360)
                                / 360;

                            out_lat.element<T>(idx) = (origin_lat - in_lat.element<T>(idx))
                                * 40000.0
                                / 360;
                         });

        return std::make_pair(std::move(output_lon),
                              std::move(output_lat));
    }
};

} // namespace anonymous

namespace cuspatial {

pair_of_columns
lonlat_to_coord(double origin_lon,
                double origin_lat,
                cudf::column_view const& input_lon,
                cudf::column_view const& input_lat,
                rmm::mr::device_memory_resource *mr)
{
    CUSPATIAL_EXPECTS(origin_lon >= -180 &&
                      origin_lon <=  180 &&
                      origin_lat >=  -90 &&
                      origin_lat <=   90,
                      "origin must have valid longitude [-180, 180] and latitude [-90, 90]");

    CUSPATIAL_EXPECTS(input_lon.size() == input_lat.size(),
                      "input x and y arrays must have the same length");

    CUSPATIAL_EXPECTS(not input_lon.has_nulls() &&
                      not input_lat.has_nulls(),
                      "input cannot contain nulls");

    cudaStream_t stream = 0;

    return cudf::experimental::type_dispatcher(input_lon.type(), ll2coord_functor(),
                                               origin_lon, origin_lat, input_lon, input_lat, mr, stream);
}

} // namespace cuspatial
