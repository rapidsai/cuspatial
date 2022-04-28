#pragma once
#include <cuspatial/types.hpp>

namespace cuspatial {

/**
 * @brief Element-wise add of two 2d vectors.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE operator+(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x + b.x, a.y + b.y};
}

/**
 * @brief Element-wise subtract of two 2d vectors.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE operator-(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x - b.x, a.y - b.y};
}

/**
 * @brief Scale a 2d vector by ratio @p r.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE operator*(vec_2d<T> vec, T const& r)
{
  return vec_2d<T>{vec.x * r, vec.y * r};
}

/**
 * @brief Scale a 2d vector by ratio @p r.
 */
template <typename T>
vec_2d<T> CUSPATIAL_HOST_DEVICE operator*(T const& r, vec_2d<T> vec)
{
  return vec * r;
}

/**
 * @brief Compute dot product of two 2d vectors.
 */
template <typename T>
T CUSPATIAL_HOST_DEVICE dot(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.x + a.y * b.y;
}

/**
 * @brief Compute 2d determinant of a 2x2 matrix with column vectors @p a and @p b.
 */
template <typename T>
T CUSPATIAL_HOST_DEVICE det(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.y - a.y * b.x;
}

}  // namespace cuspatial
