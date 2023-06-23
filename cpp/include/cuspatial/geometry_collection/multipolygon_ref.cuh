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
#include <cuspatial/detail/range/enumerate_range.cuh>

namespace cuspatial {

/**
 * @brief Represent a reference to a multipolygon stored in a structure of arrays.
 *
 * @tparam PartIterator type of iterator to the part offset array.
 * @tparam RingIterator type of iterator to the ring offset array.
 * @tparam VecIterator type of iterator to the underlying point array.
 */
template <typename PartIterator, typename RingIterator, typename VecIterator>
class multipolygon_ref {
 public:
  CUSPATIAL_HOST_DEVICE multipolygon_ref(PartIterator part_begin,
                                         PartIterator part_end,
                                         RingIterator ring_begin,
                                         RingIterator ring_end,
                                         VecIterator point_begin,
                                         VecIterator point_end);
  /// Return the number of polygons in the multipolygon.
  CUSPATIAL_HOST_DEVICE auto num_polygons() const;
  /// Return the number of polygons in the multipolygon.
  CUSPATIAL_HOST_DEVICE auto size() const { return num_polygons(); }

  /// Returns true if the multipolygon contains 0 geometries.
  CUSPATIAL_HOST_DEVICE bool is_empty() const { return num_polygons() == 0; }

  /// Return iterator to the first polygon.
  CUSPATIAL_HOST_DEVICE auto part_begin() const;
  /// Return iterator to one past the last polygon.
  CUSPATIAL_HOST_DEVICE auto part_end() const;

  /// Return iterator to the first ring.
  CUSPATIAL_HOST_DEVICE auto ring_begin() const;
  /// Return iterator to one past the last ring.
  CUSPATIAL_HOST_DEVICE auto ring_end() const;

  /// Return iterator to the first point of the multipolygon.
  CUSPATIAL_HOST_DEVICE auto point_begin() const;
  /// Return iterator to one past the last point of the multipolygon.
  CUSPATIAL_HOST_DEVICE auto point_end() const;

  /// Return iterator to the first polygon of the multipolygon.
  CUSPATIAL_HOST_DEVICE auto begin() const { return part_begin(); }
  /// Return iterator to one past the last polygon of the multipolygon.
  CUSPATIAL_HOST_DEVICE auto end() const { return part_end(); }

  /// Return an enumerated range to the polygons.
  CUSPATIAL_HOST_DEVICE auto enumerate() const { return detail::enumerate_range{begin(), end()}; }

  /// Return `polygon_idx`th polygon in the multipolygon.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto operator[](IndexType polygon_idx) const;

 protected:
  PartIterator _part_begin;
  PartIterator _part_end;
  RingIterator _ring_begin;
  RingIterator _ring_end;
  VecIterator _point_begin;
  VecIterator _point_end;
};

}  // namespace cuspatial

#include <cuspatial/detail/geometry_collection/multipolygon_ref.cuh>
