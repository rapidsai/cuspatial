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

#include <cuspatial/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <type_traits>
#include <utility>

namespace {

using pair_of_columns = std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>;

constexpr double earth_circumference_km            = 40000.0;
constexpr double earth_circumference_km_per_degree = earth_circumference_km / 360.0;
constexpr double deg_to_rad                        = M_PI / 180;

CUDA_HOST_DEVICE_CALLABLE
double midpoint(double a, double b) { return (a + b) / 2; }

CUDA_HOST_DEVICE_CALLABLE
double lon_to_x(double lon, double lat)
{
  return lon * earth_circumference_km_per_degree * cos(lat * deg_to_rad);
};

CUDA_HOST_DEVICE_CALLABLE
double lat_to_y(double lat) { return lat * earth_circumference_km_per_degree; };

struct lonlat_to_cartesian_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, pair_of_columns> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, pair_of_columns> operator()(
    double origin_lon,
    double origin_lat,
    cudf::column_view const& input_lon,
    cudf::column_view const& input_lat,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto size = input_lon.size();
    auto tid  = cudf::type_to_id<T>();
    auto type = cudf::data_type{tid};
    auto output_x =
      cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);
    auto output_y =
      cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);

    auto input_iters = thrust::make_tuple(input_lon.begin<T>(), input_lat.begin<T>());

    auto output_iters =
      thrust::make_tuple(output_x->mutable_view().begin<T>(), output_y->mutable_view().begin<T>());

    auto input_zip  = thrust::make_zip_iterator(input_iters);
    auto output_zip = thrust::make_zip_iterator(output_iters);

    auto to_cartesian = [=] __device__(auto lonlat) {
      auto lon = thrust::get<0>(lonlat);
      auto lat = thrust::get<1>(lonlat);
      return thrust::make_pair(lon_to_x(origin_lon - lon, midpoint(lat, origin_lat)),
                               lat_to_y(origin_lat - lat));
    };

    thrust::transform(
      rmm::exec_policy(stream), input_zip, input_zip + input_lon.size(), output_zip, to_cartesian);

    return std::make_pair(std::move(output_x), std::move(output_y));
  }
};

}  // namespace

namespace cuspatial {
namespace detail {

pair_of_columns lonlat_to_cartesian(double origin_lon,
                                    double origin_lat,
                                    cudf::column_view const& input_lon,
                                    cudf::column_view const& input_lat,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(input_lon.type(),
                               lonlat_to_cartesian_functor(),
                               origin_lon,
                               origin_lat,
                               input_lon,
                               input_lat,
                               stream,
                               mr);
}

}  // namespace detail

pair_of_columns lonlat_to_cartesian(double origin_lon,
                                    double origin_lat,
                                    cudf::column_view const& input_lon,
                                    cudf::column_view const& input_lat,
                                    rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(
    origin_lon >= -180 && origin_lon <= 180 && origin_lat >= -90 && origin_lat <= 90,
    "origin must have valid longitude [-180, 180] and latitude [-90, 90]");

  CUSPATIAL_EXPECTS(input_lon.size() == input_lat.size(), "inputs must have the same length");

  CUSPATIAL_EXPECTS(input_lon.type() == input_lat.type(), "inputs must have the same type");

  CUSPATIAL_EXPECTS(not input_lon.has_nulls() && not input_lat.has_nulls(),
                    "input cannot contain nulls");

  return detail::lonlat_to_cartesian(
    origin_lon, origin_lat, input_lon, input_lat, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
