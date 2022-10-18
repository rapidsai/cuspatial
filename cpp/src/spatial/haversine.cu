/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cuspatial/constants.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/haversine.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <type_traits>

namespace {

struct haversine_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("haversine_distance supports only floating-point types.");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& a_lon,
    cudf::column_view const& a_lat,
    cudf::column_view const& b_lon,
    cudf::column_view const& b_lat,
    T radius,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    if (a_lon.is_empty()) { return cudf::empty_like(a_lon); }

    auto mask_policy = cudf::mask_allocation_policy::NEVER;
    auto result      = cudf::allocate_like(a_lon, a_lon.size(), mask_policy, mr);

    auto lonlat_a = cuspatial::make_vec_2d_iterator(a_lon.begin<T>(), a_lat.begin<T>());
    auto lonlat_b = cuspatial::make_vec_2d_iterator(b_lon.begin<T>(), b_lat.begin<T>());

    cuspatial::haversine_distance(lonlat_a,
                                  lonlat_a + a_lon.size(),
                                  lonlat_b,
                                  static_cast<cudf::mutable_column_view>(*result).begin<T>(),
                                  T{radius},
                                  stream);

    return result;
  }
};

}  // anonymous namespace

namespace cuspatial {
namespace detail {

std::unique_ptr<cudf::column> haversine_distance(cudf::column_view const& a_lon,
                                                 cudf::column_view const& a_lat,
                                                 cudf::column_view const& b_lon,
                                                 cudf::column_view const& b_lat,
                                                 double radius,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(radius > 0, "radius must be positive.");

  CUSPATIAL_EXPECTS(not a_lon.has_nulls() and not a_lat.has_nulls() and not b_lon.has_nulls() and
                      not b_lat.has_nulls(),
                    "coordinates must not contain nulls.");

  CUSPATIAL_EXPECTS(
    a_lat.type() == a_lon.type() and b_lon.type() == a_lon.type() and b_lat.type() == a_lon.type(),
    "coordinates must have the same type.");

  CUSPATIAL_EXPECTS(
    a_lat.size() == a_lon.size() and b_lon.size() == a_lon.size() and b_lat.size() == a_lon.size(),
    "coordinates must have the same size.");

  return cudf::type_dispatcher(
    a_lon.type(), haversine_functor{}, a_lon, a_lat, b_lon, b_lat, radius, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> haversine_distance(cudf::column_view const& a_lon,
                                                 cudf::column_view const& a_lat,
                                                 cudf::column_view const& b_lon,
                                                 cudf::column_view const& b_lat,
                                                 double radius,
                                                 rmm::mr::device_memory_resource* mr)
{
  return cuspatial::detail::haversine_distance(
    a_lon, a_lat, b_lon, b_lat, radius, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
