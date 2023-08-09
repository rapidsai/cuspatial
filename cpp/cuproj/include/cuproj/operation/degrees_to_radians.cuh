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

#include <cuproj/constants.hpp>
#include <cuproj/detail/utility/cuda.hpp>
#include <cuproj/operation/operation.cuh>

namespace cuproj {

/**
 * @addtogroup operations
 * @{
 */

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
  CUPROJ_HOST_DEVICE Coordinate operator()(Coordinate const& coord, direction dir) const
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
  CUPROJ_HOST_DEVICE Coordinate forward(Coordinate const& coord) const
  {
    using T = typename Coordinate::value_type;
    return Coordinate{coord.x * DEG_TO_RAD<T>, coord.y * DEG_TO_RAD<T>};
  }

  /**
   * @brief Converts radians to degrees
   *
   * @param coord The coordinate to convert (lat, lon) in radians
   * @return The converted coordinate (lat, lon) in degrees
   */
  CUPROJ_HOST_DEVICE Coordinate inverse(Coordinate const& coord) const
  {
    using T = typename Coordinate::value_type;
    return Coordinate{coord.x * RAD_TO_DEG<T>, coord.y * RAD_TO_DEG<T>};
  }
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
