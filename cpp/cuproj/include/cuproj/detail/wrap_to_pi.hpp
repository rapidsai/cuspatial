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

namespace cuproj {

namespace detail {

/**
 * @brief Wrap/normalize an angle in radians to the -pi:pi range.
 *
 * @tparam T data type
 * @param longitude The angle to normalize
 * @return The normalized angle
 */
template <typename T>
CUPROJ_HOST_DEVICE T wrap_to_pi(T angle)
{
  // Let angle slightly overshoot, to avoid spurious sign switching of longitudes at the date line
  if (fabs(angle) < M_PI + EPSILON_RADIANS<T>) return angle;

  // adjust to 0..2pi range
  angle += M_PI;

  // remove integral # of 'revolutions'
  angle -= M_TWOPI<T> * floor(angle / M_TWOPI<T>);

  // adjust back to -pi..pi range
  return angle - M_PI;
}

}  // namespace detail
}  // namespace cuproj
