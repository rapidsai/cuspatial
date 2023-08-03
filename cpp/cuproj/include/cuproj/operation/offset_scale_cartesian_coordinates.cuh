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
#include <cuproj/ellipsoid.hpp>
#include <cuproj/operation/operation.cuh>
#include <cuproj/projection_parameters.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>

namespace cuproj {

/**
 * @addtogroup operations
 * @{
 */

/**
 * @brief Given Cartesian coordinates (x, y) in meters, offset and scale them
 * to the projection's origin and scale (ellipsoidal semi-major axis).
 *
 * @tparam Coordinate coordinate type
 * @tparam T coordinate value type
 */
template <typename Coordinate, typename T = typename Coordinate::value_type>
class offset_scale_cartesian_coordinates : operation<Coordinate> {
 public:
  /**
   * @brief Constructor
   *
   * @param params projection parameters, including the ellipsoid semi-major axis
   * and the projection origin
   */
  CUPROJ_HOST_DEVICE offset_scale_cartesian_coordinates(projection_parameters<T> const& params)
    : a_(params.ellipsoid_.a), ra_(T{1.0} / a_), x0_(params.x0), y0_(params.y0)
  {
  }

  /**
   * @brief Offset and scale a single coordinate
   *
   * @param coord the coordinate to offset and scale
   * @param dir the direction of the operation, either forward or inverse
   * @return the offset and scaled coordinate
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
   * @brief Scale a coordinate by the ellipsoid semi-major axis and offset it by
   * the projection origin
   *
   * @param coord the coordinate to offset and scale
   * @return the offset and scaled coordinate
   */
  CUPROJ_HOST_DEVICE Coordinate forward(Coordinate const& coord) const
  {
    return coord * a_ + Coordinate{x0_, y0_};
  };

  /**
   * @brief Offset a coordinate by the projection origin and scale it by the
   * inverse of the ellipsoid semi-major axis
   *
   * @param coord the coordinate to offset and scale
   * @return the offset and scaled coordinate
   */
  CUPROJ_HOST_DEVICE Coordinate inverse(Coordinate const& coord) const
  {
    return (coord - Coordinate{x0_, y0_}) * ra_;
  };

  T a_;   // ellipsoid semi-major axis
  T ra_;  // inverse of ellipsoid semi-major axis
  T x0_;  // projection origin x
  T y0_;  // projection origin y
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
