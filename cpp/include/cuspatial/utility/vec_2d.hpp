#pragma once
#include <cuspatial/types.hpp>

namespace cuspatial {

/**
 * @addtogroup type_definition_operator
 * @{
 * @file
 */

/**
 * @brief A 2D vector
 *
 * Generic 2d vector type.
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
struct alignas(2 * sizeof(T)) vec_2d {
  using value_type = T;
  value_type x;
  value_type y;
};

/**
 * @brief A LonLat coordinate pair
 *
 * Longitude/Latitude (LonLat) coordinate pairs. The x member represents Longitude,
 * and y represents Latitude.
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
struct alignas(2 * sizeof(T)) lonlat_2d : vec_2d<T> {
};

/**
 * @brief A Cartesian (X/Y) coordinate pair
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
struct alignas(2 * sizeof(T)) cartesian_2d : vec_2d<T> {
};

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

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
