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
#include <cuproj/operation/operation.cuh>
#include <cuproj/projection_parameters.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>

namespace cuproj {

template <typename Coordinate, typename T = typename Coordinate::value_type>
struct offset_scale_cartesian_coordinates : operation<Coordinate> {
  static constexpr T EPS_LAT      = 1e-12;
  static constexpr double M_TWOPI = 6.283185307179586476925286766559005;

  __host__ __device__ offset_scale_cartesian_coordinates(projection_parameters<T> const& params)
    : a_(params.ellipsoid_.a), ra_(T{1.0} / a_), x0_(params.x0), y0_(params.y0)
  {
  }

  // projection_parameters<T> setup(projection_parameters<T> const& params) { return params; }

  __host__ __device__ Coordinate operator()(Coordinate const& coord, direction dir) const
  {
    if (dir == direction::FORWARD)
      return forward(coord);
    else
      return inverse(coord);
  }

 private:
  __host__ __device__ Coordinate forward(Coordinate const& coord) const
  {
    return coord * a_ + Coordinate{x0_, y0_};
  };

  __host__ __device__ Coordinate inverse(Coordinate const& coord) const
  {
    return (coord - Coordinate{x0_, y0_}) * ra_;
  };

  T a_;
  T ra_;
  T x0_;
  T y0_;
};

}  // namespace cuproj
