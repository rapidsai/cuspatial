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

#include <cuproj/detail/utility/cuda.hpp>
#include <cuproj/operation/operation.cuh>

namespace cuproj {

/**
 * @addtogroup operations
 * @{
 */

/**
 * @brief Axis swap operation: swap x and y coordinates
 *
 * @tparam Coordinate the coordinate type
 */
template <typename Coordinate>
struct axis_swap : operation<Coordinate> {
  /**
   * @brief Swap x and y coordinates
   *
   * @param coord the coordinate to swap
   * @param dir (unused) the direction of the operation
   * @return the swapped coordinate
   */
  CUPROJ_HOST_DEVICE Coordinate operator()(Coordinate const& coord, direction) const
  {
    return Coordinate{coord.y, coord.x};
  }
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
