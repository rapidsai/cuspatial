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

#include <cuproj/detail/pipeline.cuh>
#include <cuproj/ellipsoid.hpp>
#include <cuproj/operation/operation.cuh>
#include <cuproj/projection_parameters.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <iterator>
#include <type_traits>

namespace cuproj {

/**
 * @addtogroup cuproj_types
 * @{
 * @file
 */

/**
 * @brief A projection object that can be invoked from `__device__` code to transform coordinates.
 *
 * @tparam Coordinate the coordinate type. This type is expected to have a `value_type` member type.
 */
template <typename Coordinate>
using device_projection = typename detail::pipeline<Coordinate>;

/**
 * @brief A projection transforms coordinates between coordinate reference systems
 *
 * Projections are constructed from a list of operations to be applied to coordinates.
 * The operations are applied in order, either forward or inverse.
 *
 * @tparam Coordinate the coordinate type
 * @tparam T the coordinate value type. Specify this if `Coordinate` does not have a `value_type`
 */
template <typename Coordinate, typename T = typename Coordinate::value_type>
class projection {
 public:
  /**
   * @brief Construct a new projection object
   *
   * @param operations the list of operations to apply to coordinates
   * @param params the projection parameters
   * @param dir the default order to execute the operations, FORWARD or INVERSE
   */
  projection(std::vector<operation_type> const& operations,
             projection_parameters<T> const& params,
             direction dir = direction::FORWARD)
    : params_(params), constructed_direction_(dir)
  {
    setup(operations);
  }

  /**
   * @brief Get a device_projection object that can be passed to device code.
   *
   * This object can be used to transform coordinates on the device.
   *
   * @note The implementation is in detail::pipeline.
   *
   * @param dir the direction of the transform, FORWARD or INVERSE.
   * @return the device projection
   */
  device_projection<Coordinate> get_device_projection(direction dir) const
  {
    dir = (constructed_direction_ == direction::FORWARD) ? dir : reverse(dir);
    return device_projection<Coordinate>{
      params_, operations_.data().get(), operations_.size(), dir};
  }

  /**
   * @brief Transform a range of coordinates
   *
   * @tparam CoordIter the coordinate iterator type
   * @param first the start of the coordinate range
   * @param last the end of the coordinate range
   * @param result the output coordinate range
   * @param dir the direction of the transform, FORWARD or INVERSE. If INVERSE, the operations will
   * run in the reverse order of the direction specified in the constructor.
   * @param stream the CUDA stream on which to run the transform
   */
  template <class InputCoordIter, class OutputCoordIter>
  void transform(InputCoordIter first,
                 InputCoordIter last,
                 OutputCoordIter result,
                 direction dir,
                 rmm::cuda_stream_view stream = rmm::cuda_stream_default) const
  {
    thrust::transform(rmm::exec_policy(stream), first, last, result, get_device_projection(dir));
  }

 private:
  void setup(std::vector<operation_type> const& operations)
  {
    std::for_each(operations.begin(), operations.end(), [&](auto const& op) {
      switch (op) {
        case operation_type::TRANSVERSE_MERCATOR: {
          auto op = transverse_mercator<Coordinate>{params_};
          params_ = op.setup(params_);
          break;
        }
        // TODO: some ops don't have setup.  Should we make them all have setup?
        default: break;
      }
    });

    operations_.resize(operations.size());
    thrust::copy(operations.begin(), operations.end(), operations_.begin());
  }

  thrust::device_vector<operation_type> operations_;
  projection_parameters<T> params_;
  direction constructed_direction_{direction::FORWARD};
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
