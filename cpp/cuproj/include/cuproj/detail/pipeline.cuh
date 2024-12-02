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

#include <cuproj/assert.cuh>
#include <cuproj/operation/axis_swap.cuh>
#include <cuproj/operation/clamp_angular_coordinates.cuh>
#include <cuproj/operation/degrees_to_radians.cuh>
#include <cuproj/operation/offset_scale_cartesian_coordinates.cuh>
#include <cuproj/operation/operation.cuh>
#include <cuproj/operation/transverse_mercator.cuh>

#include <cuda/std/iterator>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace cuproj {
namespace detail {

/**
 * @internal
 * @brief A pipeline of projection operations applied in order to a coordinate
 *
 * @tparam Coordinate the coordinate type
 * @tparam dir The direction of the pipeline, FORWARD or INVERSE
 * @tparam T the coordinate value type
 */
template <typename Coordinate, typename T = typename Coordinate::value_type>
class pipeline {
 public:
  /**
   * @brief Construct a new pipeline object with the given operations and parameters
   *
   * @param params The projection parameters
   * @param ops The operations to apply
   * @param num_stages The number of operations to apply
   */
  pipeline(projection_parameters<T> const& params,
           operation_type const* ops,
           std::size_t num_stages,
           direction dir = direction::FORWARD)
    : params_(params), d_ops(ops), num_stages(num_stages), dir_(dir)
  {
  }

  /**
   * @brief Transform a coordinate using the pipeline
   *
   * @param c The coordinate to transform
   * @return The transformed coordinate
   */
  inline __device__ Coordinate operator()(Coordinate const& c) const
  {
    Coordinate c_out{c};
    // depending on direction, get a forward or reverse iterator to d_ops
    if (dir_ == direction::FORWARD) {
      auto first = d_ops;
      thrust::for_each_n(
        thrust::seq, first, num_stages, [&](auto const& op) { c_out = dispatch_op(c_out, op); });
    } else {
      auto first = cuda::std::reverse_iterator(d_ops + num_stages);
      thrust::for_each_n(
        thrust::seq, first, num_stages, [&](auto const& op) { c_out = dispatch_op(c_out, op); });
    }
    return c_out;
  }

  /**
   * @brief Transform a coordinate using the pipeline
   *
   * @note this is an alias for operator() to allow for a more natural syntax
   *
   * @param c The coordinate to transform
   * @return The transformed coordinate
   */
  inline __device__ Coordinate transform(Coordinate const& c) const { return operator()(c); }

 private:
  projection_parameters<T> params_;
  operation_type const* d_ops;
  std::size_t num_stages;
  direction dir_;

  inline __device__ Coordinate dispatch_op(Coordinate const& c, operation_type const& op) const
  {
    switch (op) {
      case operation_type::AXIS_SWAP: {
        auto op = axis_swap<Coordinate>{};
        return op(c, dir_);
      }
      case operation_type::DEGREES_TO_RADIANS: {
        auto op = degrees_to_radians<Coordinate>{};
        return op(c, dir_);
      }
      case operation_type::CLAMP_ANGULAR_COORDINATES: {
        auto op = clamp_angular_coordinates<Coordinate>{params_};
        return op(c, dir_);
      }
      case operation_type::OFFSET_SCALE_CARTESIAN_COORDINATES: {
        auto op = offset_scale_cartesian_coordinates<Coordinate>{params_};
        return op(c, dir_);
      }
      case operation_type::TRANSVERSE_MERCATOR: {
        auto op = transverse_mercator<Coordinate>{params_};
        return op(c, dir_);
      }
      default: {
        cuproj_assert("Invalid operation type");
        return c;
      }
    }
  }
};

}  // namespace detail

}  // namespace cuproj
