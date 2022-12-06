/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/sinusoidal_projection.cuh>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <type_traits>
#include <utility>

namespace {

using pair_of_columns = std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>;

struct dispatch_sinusoidal_projection {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, pair_of_columns> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, pair_of_columns> operator()(
    T origin_lon,
    T origin_lat,
    cudf::column_view const& input_lon,
    cudf::column_view const& input_lat,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto size = input_lon.size();
    auto type = cudf::data_type{cudf::type_to_id<T>()};

    auto output_x =
      cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);
    auto output_y =
      cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);

    auto lonlat_begin = cuspatial::make_vec_2d_iterator(input_lon.begin<T>(), input_lat.begin<T>());

    auto output_zip = cuspatial::make_vec_2d_output_iterator(output_x->mutable_view().begin<T>(),
                                                             output_y->mutable_view().begin<T>());

    auto origin = cuspatial::vec_2d<T>{origin_lon, origin_lat};

    cuspatial::sinusoidal_projection(
      lonlat_begin, lonlat_begin + input_lon.size(), output_zip, origin, stream);

    return std::make_pair(std::move(output_x), std::move(output_y));
  }
};

}  // namespace

namespace cuspatial {
namespace detail {

pair_of_columns sinusoidal_projection(double origin_lon,
                                      double origin_lat,
                                      cudf::column_view const& input_lon,
                                      cudf::column_view const& input_lat,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(
    origin_lon >= -180 && origin_lon <= 180 && origin_lat >= -90 && origin_lat <= 90,
    "origin must have valid longitude [-180, 180] and latitude [-90, 90]");

  CUSPATIAL_EXPECTS(input_lon.size() == input_lat.size(), "inputs must have the same length");

  CUSPATIAL_EXPECTS(input_lon.type() == input_lat.type(), "inputs must have the same type");

  CUSPATIAL_EXPECTS(not input_lon.has_nulls() && not input_lat.has_nulls(),
                    "input cannot contain nulls");

  return cudf::type_dispatcher(input_lon.type(),
                               dispatch_sinusoidal_projection(),
                               origin_lon,
                               origin_lat,
                               input_lon,
                               input_lat,
                               stream,
                               mr);
}

}  // namespace detail

pair_of_columns sinusoidal_projection(double origin_lon,
                                      double origin_lat,
                                      cudf::column_view const& input_lon,
                                      cudf::column_view const& input_lat,
                                      rmm::mr::device_memory_resource* mr)
{
  return detail::sinusoidal_projection(
    origin_lon, origin_lat, input_lon, input_lat, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
