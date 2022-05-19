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

#include <cuspatial/utility/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <iterator>

namespace cuspatial {

/**
 * @brief Translate longitude/latitude relative to origin to cartesian (x/y) coordinates in km.
 *
 * @param[in]  lon_lat_first beginning of range of input longitude/latitude coordinates.
 * @param[in]  lon_lat_last end of range of input longitude/latitude coordinates.
 * @param[in]  origin: longitude and latitude of origin.
 * @param[out] xy_first: beginning of range of output x/y coordinates.
 * @param[in]  stream: The CUDA stream on which to perform computations and allocate memory.
 *
 * All input iterators must have a `value_type` of `cuspatial::lonlat_2d<T>` (Lat/Lon coordinates),
 * and the output iterator must be able to accept for storage values of type
 * `cuspatial::cartesian_2d<T>` (Cartesian coordinates).
 *
 * @tparam InputIt Iterator over longitude/latitude locations. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt Iterator over Cartesian output points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam T the floating-point coordinate value type of input longitude/latitude coordinates.
 *
 * @return Output iterator to the element past the last x/y coordinate computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class InputIt, class OutputIt, class T>
OutputIt lonlat_to_cartesian(InputIt lon_lat_first,
                             InputIt lon_lat_last,
                             OutputIt xy_first,
                             lonlat_2d<T> origin,
                             rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/coordinate_transform.cuh>
