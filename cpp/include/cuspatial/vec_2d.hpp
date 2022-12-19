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

#include <algorithm>
#include <ostream>

namespace cuspatial {

/**
 * @addtogroup types
 * @{
 */

/**
 * @brief A generic 2D vector type.
 *
 * This is the base type used in cuspatial for both Longitude/Latitude (LonLat) coordinate pairs and
 * Cartesian (X/Y) coordinate pairs. For LonLat pairs, the `x` member represents Longitude, and `y`
 * represents Latitude.
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
class alignas(2 * sizeof(T)) vec_2d {
 public:
  using value_type = T;
  value_type x;
  value_type y;

 private:
  /**
   * @brief Output stream operator for `vec_2d<T>` for human-readable formatting
   */
  friend std::ostream& operator<<(std::ostream& os, cuspatial::vec_2d<T> const& vec)
  {
    return os << "(" << vec.x << "," << vec.y << ")";
  }

  /**
   * @brief Compare two 2D vectors for equality.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator==(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
  {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
  }

  /**
   * @brief Element-wise addition of two 2D vectors.
   */
  friend vec_2d<T> CUSPATIAL_HOST_DEVICE operator+(vec_2d<T> const& a, vec_2d<T> const& b)
  {
    return vec_2d<T>{a.x + b.x, a.y + b.y};
  }

  /**
   * @brief Element-wise subtraction of two 2D vectors.
   */
  friend vec_2d<T> CUSPATIAL_HOST_DEVICE operator-(vec_2d<T> const& a, vec_2d<T> const& b)
  {
    return vec_2d<T>{a.x - b.x, a.y - b.y};
  }

  /**
   * @brief Invert a 2D vector.
   */
  friend vec_2d<T> CUSPATIAL_HOST_DEVICE operator-(vec_2d<T> const& a)
  {
    return vec_2d<T>{-a.x, -a.y};
  }

  /**
   * @brief Scale a 2D vector by a factor @p r.
   */
  friend vec_2d<T> CUSPATIAL_HOST_DEVICE operator*(vec_2d<T> vec, T const& r)
  {
    return vec_2d<T>{vec.x * r, vec.y * r};
  }

  /**
   * @brief Scale a 2d vector by ratio @p r.
   */
  friend vec_2d<T> CUSPATIAL_HOST_DEVICE operator*(T const& r, vec_2d<T> vec) { return vec * r; }

  /**
   * @brief Translate a 2D point
   */
  friend vec_2d<T>& CUSPATIAL_HOST_DEVICE operator+=(vec_2d<T>& a, vec_2d<T> const& b)
  {
    a.x += b.x;
    a.y += b.y;
    return a;
  }

  /**
   * @brief Translate a 2D point
   */
  friend vec_2d<T>& CUSPATIAL_HOST_DEVICE operator-=(vec_2d<T>& a, vec_2d<T> const& b)
  {
    return a += -b;
  }

  /**
   * @brief Less than operator for two 2D points.
   *
   * Orders two points first by x, then by y.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator<(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
  {
    if (lhs.x < rhs.x)
      return true;
    else if (lhs.x == rhs.x)
      return lhs.y < rhs.y;
    return false;
  }

  /**
   * @brief Greater than operator for two 2D points.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator>(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
  {
    return rhs < lhs;
  }

  /**
   * @brief Less than or equal to operator for two 2D points.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator<=(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
  {
    return !(lhs > rhs);
  }

  /**
   * @brief Greater than or equal to operator for two 2D points.
   */
  friend bool CUSPATIAL_HOST_DEVICE operator>=(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
  {
    return !(lhs < rhs);
  }
};

// Deduction guide enables CTAD
template <typename T>
vec_2d(T x, T y) -> vec_2d<T>;

/**
 * @brief Compute dot product of two 2D vectors.
 */
template <typename T>
T CUSPATIAL_HOST_DEVICE dot(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.x + a.y * b.y;
}

/**
 * @brief Compute 2D determinant of a 2x2 matrix with column vectors @p a and @p b.
 */
template <typename T>
T CUSPATIAL_HOST_DEVICE det(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.y - a.y * b.x;
}

/**
 * @brief Return a new vec_2d made up of the minimum x- and y-components of two input vec_2d values.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE box_min(vec_2d<T> const& a, vec_2d<T> const& b)
{
#ifdef __CUDA_ARCH__
  return vec_2d<T>{::min(a.x, b.x), ::min(a.y, b.y)};
#else
  return vec_2d<T>{std::min(a.x, b.x), std::min(a.y, b.y)};
#endif
}

/**
 * @brief Return a new vec_2d made up of the minimum x- and y-components of two input vec_2d values.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE box_max(vec_2d<T> const& a, vec_2d<T> const& b)
{
#ifdef __CUDA_ARCH__
  return vec_2d<T>{::max(a.x, b.x), ::max(a.y, b.y)};
#else
  return vec_2d<T>{std::max(a.x, b.x), std::max(a.y, b.y)};
#endif
}

/**
 * @brief Compute the midpoint of `first` and `second`.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE midpoint(vec_2d<T> const& first, vec_2d<T> const& second)
{
  return (first + second) * T{0.5};
}

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
