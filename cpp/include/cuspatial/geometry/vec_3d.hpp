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
#include <cuspatial/detail/utility/floating_point.cuh>

#include <algorithm>
#include <ostream>

namespace cuspatial {

/**
 * @addtogroup types
 * @{
 */

/**
 * @brief A generic 3D vector type.
 *
 * This is the base type used in cuspatial for Cartesian (X/Y/Z) coordinates.
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
class alignas(4 * sizeof(T)) vec_3d {
 public:
  using value_type = T;
  value_type x;
  value_type y;
  value_type z;

 private:
  /**
   * @brief Output stream operator for `vec_3d<T>` for human-readable formatting
   */
  friend std::ostream& operator<<(std::ostream& os, cuspatial::vec_3d<T> const& vec)
  {
    return os << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
  }

  /**
   * @brief Compare two 3D vectors for equality.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator==(vec_3d<T> const& lhs, vec_3d<T> const& rhs)
  {
    return detail::float_equal<T>(lhs.x, rhs.x) && detail::float_equal(lhs.y, rhs.y) &&
           detail::float_equal(lhs.z, rhs.z);
  }

  /**
   * @brief Element-wise addition of two 3D vectors.
   */
  friend vec_3d<T> CUSPATIAL_HOST_DEVICE operator+(vec_3d<T> const& a, vec_3d<T> const& b)
  {
    return vec_3d<T>{a.x + b.x, a.y + b.y, a.z + b.z};
  }

  /**
   * @brief Element-wise subtraction of two 3D vectors.
   */
  friend vec_3d<T> CUSPATIAL_HOST_DEVICE operator-(vec_3d<T> const& a, vec_3d<T> const& b)
  {
    return vec_3d<T>{a.x - b.x, a.y - b.y, a.z - b.z};
  }

  /**
   * @brief Invert a 3D vector.
   */
  friend vec_3d<T> CUSPATIAL_HOST_DEVICE operator-(vec_3d<T> const& a)
  {
    return vec_3d<T>{-a.x, -a.y, -a.z};
  }

  /**
   * @brief Scale a 3D vector by a factor @p r.
   */
  friend vec_3d<T> CUSPATIAL_HOST_DEVICE operator*(vec_3d<T> vec, T const& r)
  {
    return vec_3d<T>{vec.x * r, vec.y * r, vec.z * r};
  }

  /**
   * @brief Scale a 3d vector by ratio @p r.
   */
  friend vec_3d<T> CUSPATIAL_HOST_DEVICE operator*(T const& r, vec_3d<T> vec) { return vec * r; }

  /**
   * @brief Translate a 3D point
   */
  friend vec_3d<T>& CUSPATIAL_HOST_DEVICE operator+=(vec_3d<T>& a, vec_3d<T> const& b)
  {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
  }

  /**
   * @brief Translate a 3D point
   */
  friend vec_3d<T>& CUSPATIAL_HOST_DEVICE operator-=(vec_3d<T>& a, vec_3d<T> const& b)
  {
    return a += -b;
  }

  /**
   * @brief Less than operator for two 3D points.
   *
   * Orders two points first by x, then by y.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator<(vec_3d<T> const& lhs, vec_3d<T> const& rhs)
  {
    if (lhs.x < rhs.x) return true;
    if (lhs.x > rhs.x) return false;
    if (lhs.y < rhs.y) return true;
    if (lhs.y > rhs.y) return false;
    return lhs.z < rhs.z;
  }

  /**
   * @brief Greater than operator for two 3D points.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator>(vec_3d<T> const& lhs, vec_3d<T> const& rhs)
  {
    return rhs < lhs;
  }

  /**
   * @brief Less than or equal to operator for two 3D points.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator<=(vec_3d<T> const& lhs, vec_3d<T> const& rhs)
  {
    return !(lhs > rhs);
  }

  /**
   * @brief Greater than or equal to operator for two 3D points.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator>=(vec_3d<T> const& lhs, vec_3d<T> const& rhs)
  {
    return !(lhs < rhs);
  }
};

// Deduction guide enables CTAD
template <typename T>
vec_3d(T x, T y, T z) -> vec_3d<T>;

/**
 * @brief Compute dot product of two 3D vectors.
 */
template <typename T>
T CUSPATIAL_HOST_DEVICE dot(vec_3d<T> const& a, vec_3d<T> const& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Compute cross product of two 3D vectors.
 *
 * Equivalent to 3D determinant of a 2x2 matrix with column vectors @p a and @p b.
 */
template <typename T>
vec_3d<T> CUSPATIAL_HOST_DEVICE cross(vec_3d<T> const& a, vec_3d<T> const& b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

}  // namespace cuspatial
