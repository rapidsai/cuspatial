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

#include <cuspatial/cuda_utils.hpp>

namespace cuspatial {

/**
 * @brief Represent a reference to a polygon stored in a structure of arrays.
 *
 * @tparam VecIterator type of iterator to the underlying point array.
 */
template <typename RingIterator, typename VecIterator>
class polygon_ref {
 public:
  CUSPATIAL_HOST_DEVICE polygon_ref(RingIterator ring_begin,
                                    RingIterator ring_end,
                                    VecIterator point_begin,
                                    VecIterator point_end);

  /// Return the number of rings in the polygon
  CUSPATIAL_HOST_DEVICE auto num_rings() const;

  /// Return the number of rings in the polygon
  CUSPATIAL_HOST_DEVICE auto size() const { return num_rings(); }

  /// Return iterator to the first ring of the polygon
  CUSPATIAL_HOST_DEVICE auto ring_begin() const;
  /// Return iterator to one past the last ring
  CUSPATIAL_HOST_DEVICE auto ring_end() const;

  /// Return iterator to the first point of the polygon
  CUSPATIAL_HOST_DEVICE auto point_begin() const;
  /// Return iterator to one past the last point
  CUSPATIAL_HOST_DEVICE auto point_end() const;

  /// Return iterator to the first ring of the polygon
  CUSPATIAL_HOST_DEVICE auto begin() const { return ring_begin(); }
  /// Return iterator to one past the last ring
  CUSPATIAL_HOST_DEVICE auto end() const { return ring_end(); }

  /// Return the `ring_idx`th ring in the polygon.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto ring(IndexType ring_idx) const;

 protected:
  RingIterator _ring_begin;
  RingIterator _ring_end;
  VecIterator _point_begin;
  VecIterator _point_end;
};

}  // namespace cuspatial

#include <cuspatial/detail/geometry/polygon_ref.cuh>
