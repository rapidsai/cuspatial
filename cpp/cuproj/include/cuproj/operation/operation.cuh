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
#include <cuproj/projection_parameters.hpp>

namespace cuproj {

/**
 * @brief Enumerates the different types of transform operations
 *
 * This enum is used to identify the type of transform operation in the transform pipeline. Each
 * operation_type has a corresponding class that implements the operation.
 */
enum operation_type {
  AXIS_SWAP,
  DEGREES_TO_RADIANS,
  CLAMP_ANGULAR_COORDINATES,
  OFFSET_SCALE_CARTESIAN_COORDINATES,
  TRANSVERSE_MERCATOR
};

/// Enumerates the direction of a transform operation
enum direction { FORWARD, INVERSE };

/// Returns the opposite of a direction
inline direction reverse(direction dir)
{
  return dir == direction::FORWARD ? direction::INVERSE : direction::FORWARD;
}

/**
 * @brief Base class for all transform operations
 *
 * This class is used to define the interface for all transform operations. A transform operation
 * is a function object that takes a coordinate and returns a coordinate. Operations are composed
 * together to form a transform pipeline by cuproj::projection.
 *
 * @tparam Coordinate
 * @tparam Coordinate::value_type
 */
template <typename Coordinate, typename T = typename Coordinate::value_type>
class operation {
 public:
  /**
   * @brief Applies the transform operation to a coordinate
   *
   * @param c Coordinate to transform
   * @param dir Direction of transform
   * @return Coordinate
   */
  CUPROJ_HOST_DEVICE Coordinate operator()(Coordinate const& c, direction dir) const { return c; }

  /**
   * @brief Modifies the projection parameters for the transform operation
   *
   * Some (but not all) operations require additional parameters to be set in the projection_params
   * object. This function is called by cuproj::projection::setup() to allow the operation to
   * modify the parameters as needed.
   *
   * The final project_parameters are passed to every operation in the transform pipeline.
   *
   * @param params Projection parameters
   * @return The modified parameters
   */
  projection_parameters<T> setup(projection_parameters<T> const& params) { return params; };
};

}  // namespace cuproj
