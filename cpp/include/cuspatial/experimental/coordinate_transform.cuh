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

#include <cuspatial/types.hpp>

#include <iterator>

namespace cuspatial {

/**
 * @brief Translate longitude/latitude relative to origin to cartesian (x/y) coordinates in km.
 *
 * @param[in] lon_lat_first beginning of range of input longitude/latitude coordinates.
 * @param[in] lon_lat_last end of range of input longitude/latitude coordinates.
 * @param[in] origin: longitude and latitude of origin.
 * @param[out] xy_first: beginning of range of output x/y coordinates.
 *
 * All input iterators must have a `value_type` of `cuspatial::location_2d<T>`, and the output
 * iterator must have `value_type` of `cuspatial::coord_2d<T>`.
 *
 * @tparam InputIt Iterator to must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI]
 * and be device-accessible.
 * @tparam OutputIt must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-accessible.
 * @tparam Location the type of input longitude/latitude coordinates, e.g. cuspatial::location_2d<T>
 * @tparam Coordinates the type of output x/y coordinates, e.g. cuspatial::coord_2d<T>
 *
 * @return Output iterator to the element past the last x/y coordinate computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class InputIt,
          class OutputIt,
          class Location    = typename std::iterator_traits<InputIt>::value_type,
          class Coordinates = typename std::iterator_traits<OutputIt>::value_type>
OutputIt lonlat_to_cartesian(InputIt lon_lat_first,
                             InputIt lon_lat_last,
                             OutputIt xy_first,
                             Location origin);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/coordinate_transform.cuh>
