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
#include <cuspatial/cuda_utils.hpp>

namespace cuspatial {

/**
 * @brief Represent a multipoint stored in structure of array on memory.
 *
 * @tparam VecIterator type of iterator to the underlying point array.
 */
template <typename VecIterator>
class multipoint_ref {
 public:
  CUSPATIAL_HOST_DEVICE multipoint_ref(VecIterator begin, VecIterator end);

  /// Return iterator to the starting point of the multipoint.
  CUSPATIAL_HOST_DEVICE auto point_begin() const;
  /// Return iterator to one-past the last point of the multipoint.
  CUSPATIAL_HOST_DEVICE auto point_end() const;

  /// Return iterator to the starting point of the multipoint.
  CUSPATIAL_HOST_DEVICE auto begin() const { return point_begin(); }
  /// Return iterator the the one-past the last point of the multipoint.
  CUSPATIAL_HOST_DEVICE auto end() const { return point_end(); }

 protected:
  VecIterator _points_begin;
  VecIterator _points_end;
};

}  // namespace cuspatial

#include <cuspatial/experimental/detail/geometry_collection/multipoint_ref.cuh>
