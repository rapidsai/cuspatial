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

#include <cuproj/ellipsoid.hpp>

namespace cuproj {

/**
 * @addtogroup projection_parameters
 * @{
 */

/**
 * @brief Hemisphere identifier for projections
 */
enum class hemisphere { NORTH, SOUTH };

/**
 * @brief Projection parameters
 *
 * Storage for parameters for projections. This is a POD type that is passed to
 * the projection operators.
 *
 * @tparam T Coordinate value type
 */
template <typename T>
struct projection_parameters {
  projection_parameters(
    ellipsoid<T> const& e, int utm_zone, hemisphere utm_hemisphere, T lam0, T prime_meridian_offset)
    : ellipsoid_(e),
      utm_zone_(utm_zone),
      utm_hemisphere_{utm_hemisphere},
      lam0_(lam0),
      prime_meridian_offset_(prime_meridian_offset)
  {
  }

  ellipsoid<T> ellipsoid_{};                      ///< Ellipsoid parameters
  int utm_zone_{-1};                              ///< UTM zone
  hemisphere utm_hemisphere_{hemisphere::NORTH};  ///< UTM hemisphere
  T lam0_{};                                      ///< Central meridian
  T prime_meridian_offset_{};                     ///< Offset from Greenwich

  T k0{};    // scaling
  T phi0{};  // central parallel
  T x0{};    // false easting
  T y0{};    // false northing

  struct tmerc_params {
    T Qn{};      // Meridian quadrant, scaled to the projection
    T Zb{};      // Radius vector in polar coord. systems
    T cgb[6]{};  // Constants for Gauss -> Geo lat
    T cbg[6]{};  // Constants for Geo lat -> Gauss
    T utg[6]{};  // Constants for transverse Mercator -> geo
    T gtu[6]{};  // Constants for geo -> transverse Mercator
  };

  tmerc_params tmerc_params_{};
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
