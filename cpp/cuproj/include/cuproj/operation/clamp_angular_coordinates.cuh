/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuproj/detail/wrap_to_pi.hpp>
#include <cuproj/error.hpp>
#include <cuproj/operation/operation.cuh>
#include <cuproj/projection_parameters.hpp>

#include <thrust/iterator/transform_iterator.h>

namespace cuproj {

/**
 * @addtogroup operations
 * @{
 */

/**
 * @brief Clamp angular coordinates to the valid range and offset by the central meridian (lam0) and
 * an optional prime meridian offset.
 *
 * @tparam Coordinate the coordinate type
 * @tparam T the coordinate value type
 */
template <typename Coordinate, typename T = typename Coordinate::value_type>
class clamp_angular_coordinates : operation<Coordinate> {
 public:
  /**
   * @brief Construct a new clamp angular coordinates object
   *
   * @param params the projection parameters
   */
  CUPROJ_HOST_DEVICE clamp_angular_coordinates(projection_parameters<T> const& params)
    : lam0_(params.lam0_), prime_meridian_offset_(params.prime_meridian_offset_)
  {
  }

  // projection_parameters<T> setup(projection_parameters<T> const& params) { return params; }

  /**
   * @brief Clamp angular coordinate to the valid range
   *
   * @param coord The coordinate to clamp
   * @param dir The direction of the operation
   * @return The clamped coordinate
   */
  [[nodiscard]] CUPROJ_HOST_DEVICE Coordinate operator()(Coordinate const& coord,
                                                         direction dir) const
  {
    if (dir == direction::FORWARD)
      return forward(coord);
    else
      return inverse(coord);
  }

 private:
  /**
   * @brief Forward clamping operation
   *
   * Offsets the longitude by the prime meridian offset and central meridian (lam0) and clamps the
   * latitude to the range -pi/2..pi/2 radians (-90..90 degrees) and the longitude to the range
   * -pi..pi radians (-180..180 degrees).
   *
   * @param coord The coordinate to clamp
   * @return The clamped coordinate
   */
  [[nodiscard]] CUPROJ_HOST_DEVICE Coordinate forward(Coordinate const& coord) const
  {
    // check for latitude or longitude over-range
    T t = (coord.y < 0 ? -coord.y : coord.y) - M_PI_2;
    CUPROJ_HOST_DEVICE_EXPECTS(t <= EPSILON_RADIANS<T>, "Invalid latitude");
    CUPROJ_HOST_DEVICE_EXPECTS(coord.x <= 10 || coord.x >= -10, "Invalid longitude");

    Coordinate xy = coord;

    /* Clamp latitude to -pi/2..pi/2 degree range */
    auto half_pi = static_cast<T>(M_PI_2);
    xy.y         = clamp(xy.y, -half_pi, half_pi);

    // Distance from central meridian, taking system zero meridian into account
    xy.x = (xy.x - prime_meridian_offset_) - lam0_;

    // Ensure longitude is in the -pi:pi range
    xy.x = detail::wrap_to_pi(xy.x);

    return xy;
  }

  /**
   * @brief Inverse clamping operation
   *
   * Reverse-offsets the longitude by the prime meridian offset and central meridian
   * and clamps the longitude to the range -pi..pi radians (-180..180 degrees)
   *
   * @param coord The coordinate to clamp
   * @return The clamped coordinate
   */
  [[nodiscard]] inline CUPROJ_HOST_DEVICE Coordinate inverse(Coordinate const& coord) const
  {
    Coordinate xy = coord;

    // Distance from central meridian, taking system zero meridian into account
    xy.x += prime_meridian_offset_ + lam0_;

    // Ensure longitude is in the -pi:pi range
    xy.x = detail::wrap_to_pi(xy.x);

    return xy;
  }

  [[nodiscard]] inline CUPROJ_HOST_DEVICE const T& clamp(const T& val,
                                                         const T& low,
                                                         const T& high) const
  {
    CUPROJ_HOST_DEVICE_EXPECTS(!(low < high), "Invalid clamp range");
    return val < low ? low : (high < val) ? high : val;
  }

  T lam0_{};  // central meridian
  T prime_meridian_offset_{};
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
