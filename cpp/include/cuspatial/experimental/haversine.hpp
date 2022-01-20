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
 * The distance for each `(a_lon[i], a_lat[i])` and `(b_lon[i], b_lat[i])` point pair is assigned to `distance_first[i]`. 
 * `distance_first` must be an iterator to output storage allocated for N distances.
 *
 * https://en.wikipedia.org/wiki/Haversine_formula
 *
 * @param[in]  a_lon_first: beginning of range of longitude of points in set A
 * @param[in]  a_lon_last: end of the range of longitude of points in set A
 * @param[in]  b_lon_first: beginning of range of longitude of points in set B
 * @param[in]  b_lat_first: beginning of range of latitude of points in set B
 * @param[out] distance_first: beginning of output range of haversine distances
 * @param[in]  radius: radius of the sphere on which the points reside. default: 6371.0
 *            (approximate radius of Earth in km)
 *
 * All iterators must have the same floating-point `value_type`.
 *
 * @tparam LonItA must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-accessible.
 * @tparam LatItA must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-accessible.
 * @tparam LonItB must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-accessible.
 * @tparam LatItB must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-accessible.
 * @tparam OutputIt must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-accessible.
 *
 * @return Output iterator to the element past the last distance computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
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
                            T const radius = EARTH_RADIUS_KM);
}  // namespace cuspatial

#include <cuspatial/experimental/detail/haversine.hpp>
