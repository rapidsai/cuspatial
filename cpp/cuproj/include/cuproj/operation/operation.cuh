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

#include <cuproj/projection_parameters.hpp>

namespace cuproj {

enum operation_type {
  AXIS_SWAP,
  DEGREES_TO_RADIANS,
  CLAMP_ANGULAR_COORDINATES,
  OFFSET_SCALE_CARTESIAN_COORDINATES,
  TRANSVERSE_MERCATOR
};

enum class direction { FORWARD, INVERSE };

// base class for all operations
template <typename Coordinate, typename T = typename Coordinate::value_type>
struct operation {
  __host__ __device__ Coordinate operator()(Coordinate const& c, direction dir) const { return c; }

  __host__ projection_parameters<T> setup(projection_parameters<T> const& params)
  {
    return params;
  };
};

}  // namespace cuproj
