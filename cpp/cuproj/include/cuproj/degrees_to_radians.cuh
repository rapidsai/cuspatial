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

#include <cuproj/projection.cuh>

namespace cuproj {

constexpr double DEG_TO_RAD = 0.017453292519943295769236907684886;

template <typename Coordinate>
struct degrees_to_radians : operation<Coordinate> {
  __host__ __device__ Coordinate operator()(Coordinate const& coord) const override
  {
    using T = typename Coordinate::value_type;
    return Coordinate{static_cast<T>(coord.x * DEG_TO_RAD), static_cast<T>(coord.y * DEG_TO_RAD)};
  }
};

}  // namespace cuproj
