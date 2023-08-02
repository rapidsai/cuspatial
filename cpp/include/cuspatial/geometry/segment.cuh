/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cuspatial/geometry/vec_2d.hpp>

#include <thrust/device_reference.h>

#include <iostream>
namespace cuspatial {

/**
 * @addtogroup types
 * @{
 */

/**
 * @brief A generic segment type.
 *
 * @tparam T the base type for the coordinates
 * @tparam Vertex the vector type to use for vertices, vec_2d<T> by default
 */

template <typename T, typename Vertex = cuspatial::vec_2d<T>>
class alignas(sizeof(Vertex)) segment {
 public:
  using value_type = T;
  Vertex v1;
  Vertex v2;

  /// Return a copy of segment, translated by `v`.
  segment<T> CUSPATIAL_HOST_DEVICE translate(Vertex const& v) const
  {
    return segment<T>{v1 + v, v2 + v};
  }

  /// Return the geometric center of segment.
  Vertex CUSPATIAL_HOST_DEVICE center() const { return midpoint(v1, v2); }

  /// Return the length squared of segment.
  T CUSPATIAL_HOST_DEVICE length2() const { return dot(v2 - v1, v2 - v1); }

  /// Return slope of segment.
  T CUSPATIAL_HOST_DEVICE slope() { return (v2.y - v1.y) / (v2.x - v1.x); }

  /// Return the lower left vertex of segment.
  Vertex CUSPATIAL_HOST_DEVICE lower_left() { return v1 < v2 ? v1 : v2; }

  /// Returns true if two segments are on the same line
  /// Test if the determinant of the matrix, of which the column vector is constructed from the
  /// segments is 0.
  bool CUSPATIAL_HOST_DEVICE collinear(segment<T> const& other)
  {
    return (v1.x - v2.x) * (other.v1.y - other.v2.y) == (v1.y - v2.y) * (other.v1.x - other.v2.x);
  }

 private:
  friend std::ostream& operator<<(std::ostream& os, segment<T> const& seg)
  {
    return os << seg.v1 << " -> " << seg.v2;
  }
};

// deduction guide, enables CTAD
template <typename T>
segment(vec_2d<T> a, vec_2d<T> b) -> segment<T, vec_2d<T>>;

template <typename T>
segment(thrust::device_reference<vec_2d<T>> a, thrust::device_reference<vec_2d<T>> b)
  -> segment<T, vec_2d<T>>;
}  // namespace cuspatial
