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

#include <cuproj/operation.cuh>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>

namespace cuproj {

template <typename Coordinate, typename T = typename Coordinate::value_type>
struct clamp_angular_coordinates : operation<Coordinate> {
  static constexpr T EPS_LAT      = 1e-12;
  static constexpr double M_TWOPI = 6.283185307179586476925286766559005;

  clamp_angular_coordinates(T lam0, T prime_meridian_offset)
    : lam0_(lam0), prime_meridian_offset_(prime_meridian_offset)
  {
  }

  __host__ __device__ Coordinate operator()(Coordinate const& coord) const override
  {
    // check for latitude or longitude over-range
    T t = (coord.y < 0 ? -coord.y : coord.y) - M_PI_2;

    // TODO use host-device assert
    // CUPROJ_EXPECTS(t <= EPS_LAT, "Invalid latitude");
    // CUPROJ_EXPECTS(coord.x <= 10 || coord.x >= -10, "Invalid longitude");

    Coordinate xy = coord;

    /* Clamp latitude to -90..90 degree range */
    auto half_pi = static_cast<T>(M_PI_2);
    xy.y         = std::clamp(xy.y, -half_pi, half_pi);

    // Ensure longitude is in the -pi:pi range
    xy.x = clamp_longitude(xy.x);

    // Distance from central meridian, taking system zero meridian into account
    xy.x = (xy.x - prime_meridian_offset_) - lam0_;

    // Ensure longitude is in the -pi:pi range
    xy.x = clamp_longitude(xy.x);

    return xy;
  }

  T lam0_;
  T prime_meridian_offset_;

 private:
  __host__ __device__ T clamp_longitude(T longitude) const
  {
    // Let longitude slightly overshoot, to avoid spurious sign switching at the date line
    if (fabs(longitude) < M_PI + 1e-12) return longitude;

    // adjust to 0..2pi range
    longitude += M_PI;

    // remove integral # of 'revolutions'
    longitude -= M_TWOPI * floor(longitude / M_TWOPI);

    // adjust back to -pi..pi range
    return longitude - M_PI;
  }
};

}  // namespace cuproj
