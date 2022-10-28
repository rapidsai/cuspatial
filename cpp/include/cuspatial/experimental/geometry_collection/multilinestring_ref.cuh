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
 * @brief Represent a multilinestring stored in structure of array on memory.
 *
 * @tparam VecIterator type of iterator to the underlying point array.
 */
template <typename PartIterator, typename VecIterator>
class multilinestring_ref {
 public:
  CUSPATIAL_HOST_DEVICE multilinestring_ref(PartIterator part_begin,
                                            PartIterator part_end,
                                            VecIterator point_begin,
                                            VecIterator point_end);

  CUSPATIAL_HOST_DEVICE auto num_linestrings() const;
  CUSPATIAL_HOST_DEVICE auto size() const { return num_linestrings(); }

  /// Return iterator to the starting linestring.
  CUSPATIAL_HOST_DEVICE auto part_begin() const;
  /// Return iterator to one-past the last linestring.
  CUSPATIAL_HOST_DEVICE auto part_end() const;

  /// Return iterator to the starting point of the multipoint.
  CUSPATIAL_HOST_DEVICE auto point_begin() const;
  /// Return iterator to one-past the last point of the multipoint.
  CUSPATIAL_HOST_DEVICE auto point_end() const;

  /// Return iterator to the starting point of the multipoint.
  CUSPATIAL_HOST_DEVICE auto begin() const { return part_begin(); }
  /// Return iterator the the one-past the last point of the multipoint.
  CUSPATIAL_HOST_DEVICE auto end() const { return part_end(); }

 protected:
  PartIterator _part_begin;
  PartIterator _part_end;
  VecIterator _point_begin;
  VecIterator _point_end;
};

}  // namespace cuspatial

#include <cuspatial/experimental/detail/geometry_collection/multilinestring_ref.cuh>
