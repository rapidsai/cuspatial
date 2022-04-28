#pragma once
#include <cuspatial/types.hpp>

namespace cuspatial {

template <typename T>
vec_2d<T> __device__ operator+(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x + b.x, a.y + b.y};
}

template <typename T>
vec_2d<T> __device__ operator-(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x - b.x, a.y - b.y};
}

template <typename T>
vec_2d<T> __device__ operator*(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x * b.x, a.y * b.y};
}

template <typename T>
vec_2d<T> __device__ operator*(vec_2d<T> vec, T const& r)
{
  return vec_2d<T>{vec.x * r, vec.y * r};
}

template <typename T>
vec_2d<T> __device__ operator*(T const& r, vec_2d<T> vec)
{
  return vec * r;
}

template <typename T>
T __device__ dot(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.x + a.y * b.y;
}

template <typename T>
T __device__ det(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.y - a.y * b.x;
}

}  // namespace cuspatial
