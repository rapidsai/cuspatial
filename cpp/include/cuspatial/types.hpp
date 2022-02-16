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

#pragma once

#ifdef __CUDACC__
#define CUSPATIAL_HOST_DEVICE __host__ __device__
#else
#define CUSPATIAL_HOST_DEVICE
#endif

#include <cstdint>

namespace cuspatial {

/**
 * @brief A 2D location: latitude and longitude
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
struct alignas(2 * sizeof(T)) location_2d {
  using value_type = T;
  value_type longitude;
  value_type latitude;
};

/**
 * @brief A timestamp
 *
 */
struct its_timestamp {
  std::uint32_t y   : 6;
  std::uint32_t m   : 4;
  std::uint32_t d   : 5;
  std::uint32_t hh  : 5;
  std::uint32_t mm  : 6;
  std::uint32_t ss  : 6;
  std::uint32_t wd  : 3;
  std::uint32_t yd  : 9;
  std::uint32_t ms  : 10;
  std::uint32_t pid : 10;
};

}  // namespace cuspatial
