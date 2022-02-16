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

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace cuspatial {

/**
 * brief Compute haversine distances between points in set A to the corresponding points in set B.
 *
 * Computes N haversine distances, where N is `std::distance(a_lon_first, a_lon_last)`.
 * The distance for each `(a_lon[i], a_lat[i])` and `(b_lon[i], b_lat[i])` point pair is assigned to
 * `distance_first[i]`. `distance_first` must be an iterator to output storage allocated for N
 * distances.
 *
 * https://en.wikipedia.org/wiki/Haversine_formula
 *
 * @param[in]  a_lonlat_first: beginning of range of (longitude, latitude) locations in set A
 * @param[in]  a_lonlat_last: end of range of (longitude, latitude) locations in set A
 * @param[in]  b_lonlat_first: beginning of range of (longitude, latitude) locations in set B
 * @param[out] distance_first: beginning of output range of haversine distances
 * @param[in]  radius: radius of the sphere on which the points reside. default: 6371.0
 *            (approximate radius of Earth in km)
 *
 * All iterators must have the same floating-point `value_type`.
 *
 * Computed distances will have the same units as `radius`.
 *
 * @tparam LonLatItA Iterator to input location set A. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam LonLatItB Iterator to input location set B. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt Output iterator. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam Location The `value_type` of `LonLatItA` and `LonLatItB`. Must be
 * `cuspatial::location_2d<T>`.
 * @tparam T The underlying coordinate type. Must be a floating-point type.
 *
 * @return Output iterator to the element past the last distance computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class LonLatItA,
          class LonLatItB,
          class OutputIt,
          class Location = typename std::iterator_traits<LonLatItA>::value_type,
          class T        = typename Location::value_type>
OutputIt haversine_distance(LonLatItA a_lonlat_first,
                            LonLatItA a_lonlat_last,
                            LonLatItB b_lonlat_first,
                            OutputIt distance_first,
                            T const radius = EARTH_RADIUS_KM);
}  // namespace cuspatial

#include <cuspatial/experimental/detail/haversine.hpp>
