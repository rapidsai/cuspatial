/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <cuspatial/constants.hpp>
#include <cuspatial/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cuspatial {

namespace detail {

template <typename T>
__device__ T calculate_haversine_distance(T radius, T a_lon, T a_lat, T b_lon, T b_lat)
{
  auto ax = a_lon * DEGREE_TO_RADIAN;
  auto ay = a_lat * DEGREE_TO_RADIAN;
  auto bx = b_lon * DEGREE_TO_RADIAN;
  auto by = b_lat * DEGREE_TO_RADIAN;

  // haversine formula
  auto x        = (bx - ax) / 2;
  auto y        = (by - ay) / 2;
  auto sinysqrd = sin(y) * sin(y);
  auto sinxsqrd = sin(x) * sin(x);
  auto scale    = cos(ay) * cos(by);

  return 2 * radius * asin(sqrt(sinysqrd + sinxsqrd * scale));
};

template <class LonItA,
          class LatItA,
          class LonItB,
          class LatItB,
          class OutputIt,
          class T = typename std::iterator_traits<LonItA>::value_type>
OutputIt haversine_distance(LonItA a_lon_first,
                            LonItA a_lon_last,
                            LatItA a_lat_first,
                            LonItB b_lon_first,
                            LatItB b_lat_first,
                            OutputIt distance_first,
                            T const radius               = EARTH_RADIUS_KM,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  static_assert(
    std::conjunction_v<std::is_floating_point<T>,
                       std::is_floating_point<typename std::iterator_traits<LonItA>::value_type>,
                       std::is_floating_point<typename std::iterator_traits<LatItA>::value_type>,
                       std::is_floating_point<typename std::iterator_traits<LonItB>::value_type>,
                       std::is_floating_point<typename std::iterator_traits<LatItB>::value_type>,
                       std::is_floating_point<typename std::iterator_traits<OutputIt>::value_type>>,
    "Haversine distance supports only floating-point coordinates.");

  CUSPATIAL_EXPECTS(radius > 0, "radius must be positive.");

  auto input_tuple = thrust::make_tuple(thrust::make_constant_iterator(static_cast<T>(radius)),
                                        a_lon_first,
                                        a_lat_first,
                                        b_lon_first,
                                        b_lat_first);

  auto input_iter = thrust::make_zip_iterator(input_tuple);

  auto output_size = std::distance(a_lon_first, a_lon_last);

  return thrust::transform(rmm::exec_policy(stream),
                           input_iter,
                           input_iter + output_size,
                           distance_first,
                           [] __device__(auto inputs) {
                             return calculate_haversine_distance(thrust::get<0>(inputs),
                                                                 thrust::get<1>(inputs),
                                                                 thrust::get<2>(inputs),
                                                                 thrust::get<3>(inputs),
                                                                 thrust::get<4>(inputs));
                           });
}
}  // namespace detail

template <class LonItA, class LatItA, class LonItB, class LatItB, class OutputIt, class T>
OutputIt haversine_distance(LonItA a_lon_first,
                            LonItA a_lon_last,
                            LatItA a_lat_first,
                            LonItB b_lon_first,
                            LatItB b_lat_first,
                            OutputIt distance_first,
                            T const radius)
{
  return detail::haversine_distance(a_lon_first,
                                    a_lon_last,
                                    a_lat_first,
                                    b_lon_first,
                                    b_lat_first,
                                    distance_first,
                                    radius,
                                    rmm::cuda_stream_default);
}

}  // namespace cuspatial
