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
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <iterator>

namespace cuspatial {

/**
 * @brief Compute haversine distances between points in set A to the corresponding points in set B.
 *
 * @ingroup distance
 *
 * Computes N haversine distances, where N is `std::distance(a_lonlat_first, a_lonlat_last)`.
 * The distance for each `a_lonlat[i]` and `b_lonlat[i]` point pair is assigned to
 * `distance_first[i]`. `distance_first` must be an iterator to output storage allocated for N
 * distances.
 *
 * Computed distances will have the same units as `radius`.
 *
 * https://en.wikipedia.org/wiki/Haversine_formula
 *
 * @param[in]  a_lonlat_first: beginning of range of (longitude, latitude) locations in set A
 * @param[in]  a_lonlat_last: end of range of (longitude, latitude) locations in set A
 * @param[in]  b_lonlat_first: beginning of range of (longitude, latitude) locations in set B
 * @param[out] distance_first: beginning of output range of haversine distances
 * @param[in]  radius: radius of the sphere on which the points reside. default: 6371.0
 *            (approximate radius of Earth in km)
 * @param[in]  stream: The CUDA stream on which to perform computations and allocate memory.
 *
 * @tparam LonLatItA Iterator to input location set A. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam LonLatItB Iterator to input location set B. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt Output iterator. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible and mutable.
 * @tparam T The underlying coordinate type. Must be a floating-point type.
 *
 * @pre All iterators must have the same `Location` type, with  the same underlying floating-point
 * coordinate type (e.g. `cuspatial::vec_2d<float>`).
 *
 * @return Output iterator to the element past the last distance computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class LonLatItA,
          class LonLatItB,
          class OutputIt,
          class T = typename cuspatial::iterator_vec_base_type<LonLatItA>>
OutputIt haversine_distance(LonLatItA a_lonlat_first,
                            LonLatItA a_lonlat_last,
                            LonLatItB b_lonlat_first,
                            OutputIt distance_first,
                            T const radius               = EARTH_RADIUS_KM,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/haversine.cuh>
