/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuproj/operation/operation.cuh>

namespace cuproj {

constexpr double DEG_TO_RAD = 0.017453292519943295769236907684886;
constexpr double RAD_TO_DEG = 57.295779513082320876798154814105;

/**
 * @brief Converts degrees to radians and vice versa
 *
 * @tparam Coordinate The coordinate type
 */
template <typename Coordinate>
class degrees_to_radians : operation<Coordinate> {
 public:
  /**
   * @brief Converts degrees to radians and vice versa
   *
   * @param coord The coordinate to convert
   * @param dir The direction of the conversion: FORWARD converts degrees to radians, INVERSE
   * converts radians to degrees
   * @return The converted coordinate
   */
  __host__ __device__ Coordinate operator()(Coordinate const& coord, direction dir) const
  {
    if (dir == direction::FORWARD)
      return forward(coord);
    else
      return inverse(coord);
  }

 private:
  /**
   * @brief Converts degrees to radians
   *
   * @param coord The coordinate to convert (lat, lon) in degrees
   * @return The converted coordinate (lat, lon) in radians
   */
  __host__ __device__ Coordinate forward(Coordinate const& coord) const
  {
    using T = typename Coordinate::value_type;
    return Coordinate{static_cast<T>(coord.x * DEG_TO_RAD), static_cast<T>(coord.y * DEG_TO_RAD)};
  }

  /**
   * @brief Converts radians to degrees
   *
   * @param coord The coordinate to convert (lat, lon) in radians
   * @return The converted coordinate (lat, lon) in degrees
   */
  __host__ __device__ Coordinate inverse(Coordinate const& coord) const
  {
    using T = typename Coordinate::value_type;
    return Coordinate{static_cast<T>(coord.x * RAD_TO_DEG), static_cast<T>(coord.y * RAD_TO_DEG)};
  }
};

}  // namespace cuproj