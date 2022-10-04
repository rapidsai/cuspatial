/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial/experimental/geometry_collection/multipoint.cuh>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/traits.hpp>

namespace cuspatial {
namespace array {

using namespace cuspatial::geometry_collection;

template <typename GeometryIterator, typename VecIterator>
class multipoint_array {
 public:
  using element_t = iterator_vec_base_type<VecIterator>;

  /**
   * @brief Construct a new multipoint array object
   */
  multipoint_array(GeometryIterator geometry_begin,
                   GeometryIterator geometry_end,
                   VecIterator points_begin,
                   VecIterator points_end);

  /**
   * @brief Returns the number of multipoints in the array.
   */
  auto size();

  /**
   * @brief Returns the iterator to the start of the multipoint array.
   */
  auto multipoint_begin();

  /**
   * @brief Returns the iterator to the end of the multipoint array.
   */
  auto multipoint_end();

  /**
   * @brief Returns the iterator to the start of the underlying point array.
   */
  auto point_begin();

  /**
   * @brief Returns the iterator to the end of the underlying point array.
   */
  auto point_end();

  /**
   * @brief Returns the `idx`th multipoint in the array.
   *
   * @tparam IndexType type of the index
   * @param idx the index to the multipoint
   * @return a multipoint object
   */
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto element(IndexType idx);

 protected:
  /// Iterator to the start of the index array of start positions to each multipoint.
  GeometryIterator _geometry_begin;
  /// Iterator to the past-the-end of the index array of start positions to each multipoint.
  GeometryIterator _geometry_end;
  /// Iterator to the start of the point array.
  VecIterator _points_begin;
  /// Iterator to the past-the-end position of the point array.
  VecIterator _points_end;
};

}  // namespace array
}  // namespace cuspatial

#include <cuspatial/experimental/detail/array/multipoint_array.cuh>
