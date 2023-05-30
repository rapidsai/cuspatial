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

#include <cuspatial/error.hpp>
#include <cuspatial/geometry/box.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/traits.hpp>

#include <thrust/binary_search.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief Helper to create a `transform_iterator` that transforms sequential values.
 */
template <typename IndexType, typename UnaryFunction>
inline CUSPATIAL_HOST_DEVICE auto make_counting_transform_iterator(IndexType start, UnaryFunction f)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator(start), f);
}

/**
 * @internal
 * @brief Helper to convert a tuple of elements into a `vec_2d`
 */
template <typename T, typename VectorType = vec_2d<T>>
struct tuple_to_vec_2d {
  __device__ VectorType operator()(thrust::tuple<T, T> const& pos)
  {
    return VectorType{thrust::get<0>(pos), thrust::get<1>(pos)};
  }
};

/**
 * @internal
 * @brief Helper to convert a `vec_2d` into a tuple of elements
 */
template <typename T, typename VectorType = vec_2d<T>>
struct vec_2d_to_tuple {
  __device__ thrust::tuple<T, T> operator()(VectorType const& xy)
  {
    return thrust::make_tuple(xy.x, xy.y);
  }
};

/**
 * @internal
 * @brief Helper to convert a `box` into a tuple of elements
 */
template <typename T, typename Box = box<T>>
struct box_to_tuple {
  __device__ thrust::tuple<T, T, T, T> operator()(Box const& box)
  {
    return thrust::make_tuple(box.v1.x, box.v1.y, box.v2.x, box.v2.y);
  }
};

/**
 * @internal
 * @brief Helper to convert a tuple of vec_2d into a `box`
 */
template <typename T, typename Vertex = vec_2d<T>>
struct vec_2d_tuple_to_box {
  __device__ box<T, Vertex> operator()(thrust::tuple<Vertex, Vertex> pair)
  {
    auto v1 = thrust::get<0>(pair);
    auto v2 = thrust::get<1>(pair);
    return {v1, v2};
  }
};

/**
 * @internal
 * @brief Generic to convert any iterator pointing to interleaved xy range into
 * iterator of vec_2d.
 *
 * This generic version does not use of vectorized load.
 *
 * @pre `Iter` has operator[] defined.
 * @pre std::iterator_traits<Iter>::value_type is convertible to `T`.
 */
template <typename Iter, typename Enable = void>
struct interleaved_to_vec_2d {
  using element_t  = typename std::iterator_traits<Iter>::value_type;
  using value_type = vec_2d<element_t>;
  Iter it;
  constexpr interleaved_to_vec_2d(Iter it) : it{it} {}

  CUSPATIAL_HOST_DEVICE value_type operator()(std::size_t i)
  {
    return vec_2d<element_t>{it[2 * i], it[2 * i + 1]};
  }
};

/**
 * @brief Specialization for thrust iterators conforming to `contiguous_iterator`. (including raw
 * pointer)
 *
 * This iterator specific version uses vectorized load.
 *
 * @throw cuspatial::logic_error if `Iter` is not aligned to type `vec_2d<T>`
 * @pre `Iter` is a `contiguous_iterator` (including raw pointer).
 */
template <typename Iter>
struct interleaved_to_vec_2d<Iter,
                             typename std::enable_if_t<thrust::is_contiguous_iterator_v<Iter>>> {
  using element_t  = typename std::iterator_traits<Iter>::value_type;
  using value_type = vec_2d<element_t>;

  element_t const* ptr;

  constexpr interleaved_to_vec_2d(Iter it) : ptr{&thrust::raw_reference_cast(*it)}
  {
    CUSPATIAL_EXPECTS(!((intptr_t)ptr % alignof(vec_2d<element_t>)),
                      "Misaligned interleaved data.");
  }

  CUSPATIAL_HOST_DEVICE value_type operator()(std::size_t i)
  {
    auto const aligned =
      static_cast<element_t const*>(__builtin_assume_aligned(ptr + 2 * i, 2 * sizeof(element_t)));
    return vec_2d<element_t>{aligned[0], aligned[1]};
  }
};

/**
 * @internal
 * @brief Functor to transform an index to strided index.
 */
template <int stride>
struct strided_functor {
  auto __device__ operator()(std::size_t i) { return i * stride; }
};

/**
 * @internal
 * @brief Functor to transform an index into a geometry ID determined by a range of offsets.
 */
template <class IndexT, class GeometryIter>
struct index_to_geometry_id {
  GeometryIter geometry_begin;
  GeometryIter geometry_end;

  index_to_geometry_id(GeometryIter begin, GeometryIter end)
    : geometry_begin(begin), geometry_end(end)
  {
  }

  CUSPATIAL_HOST_DEVICE auto operator()(IndexT idx)
  {
    return thrust::distance(geometry_begin,
                            thrust::upper_bound(thrust::seq, geometry_begin, geometry_end, idx));
  }
};

/**
 * @brief Given iterator a pair of offsets, return the number of elements between the offsets.
 *
 * Used to create iterator to geometry counts, such as `multi*_point_count_begin`,
 * `multi*_segment_count_begin`.
 *
 * Example:
 * pair of offsets: (0, 3), (3, 5), (5, 8)
 * number of elements between offsets: 3, 2, 3
 *
 * @tparam OffsetPairIterator Must be iterator type to thrust::pair of indices.
 * @param p Iterator of thrust::pair of indices.
 */
struct offset_pair_to_count_functor {
  template <typename OffsetPairIterator>
  CUSPATIAL_HOST_DEVICE auto operator()(OffsetPairIterator p)
  {
    return thrust::get<1>(p) - thrust::get<0>(p);
  }
};

}  // namespace detail
}  // namespace cuspatial
