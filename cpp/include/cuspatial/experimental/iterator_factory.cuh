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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/geometry/box.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <thrust/binary_search.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <type_traits>

namespace cuspatial {
namespace detail {
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
    return thrust::distance(
      geometry_begin,
      thrust::prev(thrust::upper_bound(thrust::seq, geometry_begin, geometry_end, idx)));
  }
};

}  // namespace detail

/**
 * @addtogroup type_factories
 * @{
 */

/**
 * @brief Create an iterator to `vec_2d` data from two input iterators.
 *
 * Interleaves x and y coordinates from separate iterators into a single iterator to xy-
 * coordinates.
 *
 * @tparam VectorType cuSpatial vector type, must be `vec_2d`
 * @tparam FirstIter Iterator type to the first component of `vec_2d`. Must meet the requirements
 * of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam SecondIter Iterator type to the second component of `vec_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @param first Iterator to beginning of `vec_2d::x`
 * @param second Iterator to beginning of `vec_2d::y`
 * @return Iterator to `vec_2d`
 *
 * @pre `first` and `second` must iterate on same data type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename FirstIter, typename SecondIter>
auto make_vec_2d_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  static_assert(is_same<T, iterator_value_type<SecondIter>>(), "Iterator value_type mismatch");

  auto zipped = thrust::make_zip_iterator(first, second);
  return thrust::make_transform_iterator(zipped, detail::tuple_to_vec_2d<T>());
}

/**
 * @brief Create an iterator to `vec_2d` data from a single iterator.
 *
 * Creates a vec2d view from an interator to the starting range of interleaved x-y coordinates.
 *
 * @tparam
 * @param d_points_begin
 * @return
 */
template <typename Iter>
auto make_vec_2d_iterator(Iter xy_begin)
{
  return detail::make_counting_transform_iterator(0, detail::interleaved_to_vec_2d<Iter>{xy_begin});
}

/**
 * @brief Create an output iterator to `vec_2d` data from two output iterators.
 *
 * Creates an output iterator from separate iterators to x and y data to which
 * can be written interleaved x/y data. This allows using two separate arrays of
 * output data with APIs that expect an iterator to structured data.
 *
 * @tparam VectorType cuSpatial vector type, must be `vec_2d`
 * @tparam FirstIter Iterator type to the first component of `vec_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @tparam SecondIter Iterator type to the second component of `vec_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @param first Iterator to beginning of `x` data.
 * @param second Iterator to beginning of `y` data.
 * @return Iterator to `vec_2d`
 *
 * @pre `first` and `second` must iterate on same data type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename FirstIter, typename SecondIter>
auto make_vec_2d_output_iterator(FirstIter first, SecondIter second)
{
  using T         = typename std::iterator_traits<FirstIter>::value_type;
  auto zipped_out = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::transform_output_iterator(zipped_out, detail::vec_2d_to_tuple<T>());
}

/**
 * @brief Create an output iterator to `vec_2d` data from an iterator to an interleaved array.
 *
 * @tparam Iter type of iterator to interleaved data. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @param d_points_begin Iterator to beginning of interleaved data.
 * @return Iterator to `vec_2d`
 */
template <typename Iter>
auto make_vec_2d_output_iterator(Iter d_points_begin)
{
  using T                     = typename std::iterator_traits<Iter>::value_type;
  auto fixed_stride_2_functor = detail::strided_functor<2>();
  auto even_positions         = thrust::make_permutation_iterator(
    d_points_begin, detail::make_counting_transform_iterator(0, fixed_stride_2_functor));
  auto odd_positions = thrust::make_permutation_iterator(
    thrust::next(d_points_begin),
    detail::make_counting_transform_iterator(0, fixed_stride_2_functor));
  auto zipped_outputs =
    thrust::make_zip_iterator(thrust::make_tuple(even_positions, odd_positions));
  return thrust::make_transform_output_iterator(zipped_outputs, detail::vec_2d_to_tuple<T>());
}

/**
 * @brief Create an iterator to `box` data from two input iterators of `vec_2d`.
 *
 * Interleaves box_min and box_max points from separate iterators into a single iterator of `box`.
 *
 * @tparam VectorType cuSpatial vector type, must be `vec_2d`
 * @tparam FirstIter Iterator of `vec_2d`. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam SecondIter Iterator of `vec_2d`. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @param first Iterator to beginning of `box::v1`
 * @param second Iterator to beginning of `box::v2`
 * @return Iterator to `box`
 *
 * @pre `first` and `second` must iterate the same vec_2d data type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename FirstIter, typename SecondIter>
auto make_box_iterator(FirstIter first, SecondIter second)
{
  using Vertex = typename cuspatial::iterator_value_type<FirstIter>;
  using T      = typename Vertex::value_type;

  static_assert(is_same<Vertex, cuspatial::iterator_value_type<SecondIter>>(),
                "Iterator value_type mismatch");

  auto zipped = thrust::make_zip_iterator(first, second);
  return thrust::make_transform_iterator(zipped, detail::vec_2d_tuple_to_box<T>());
}

/**
 * @brief Create an output iterator to `box` data from multiple output iterators.
 *
 * Creates an output iterator from separate coordinate iterators to which
 * can be written interleaved x/y data for box vertices (2 per box). This allows
 * using four separate arrays of output data with APIs that expect an iterator to
 * structured box data.
 *
 * @tparam MinXIter Iterator type to the x-coordinate of the first box vertex. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @tparam MinYIter Iterator type to the y-coordinate of the first box vertex. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @tparam MaxXIter Iterator type to the x-coordinate of the second box vertex. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @tparam MaxYIter Iterator type to the y-coordinate of the second box vertex. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @param min_x Iterator to beginning of `x` data for first box vertices.
 * @param min_y Iterator to beginning of `y` data for first box vertices.
 * @param max_x Iterator to beginning of `x` data for second box vertices.
 * @param max_y Iterator to beginning of `y` data for second box vertices.
 * @return Iterator to `box`
 *
 * @pre Input iterators must iterate on the same data type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename MinXIter, typename MinYIter, typename MaxXIter, typename MaxYIter>
auto make_box_output_iterator(MinXIter min_x, MinYIter min_y, MaxXIter max_x, MaxYIter max_y)
{
  using T         = typename std::iterator_traits<MinXIter>::value_type;
  auto zipped_out = thrust::make_zip_iterator(min_x, min_y, max_x, max_y);
  return thrust::transform_output_iterator(zipped_out, detail::box_to_tuple<T>());
}

/**
 * @brief Create an input iterator that generates zero-based sequential geometry IDs for each
 * element based on the input offset range.
 *
 * This can be used for any single-level geometry offsets, e.g. multipoints, multilinestrings,
 * (multi)trajectories. And using custom iterators it can be used for nested types.
 *
 * Example:
 * @code
 * auto offsets    = std::vector<int>({0, 3, 5, 9});
 * auto iter_first = make_geometry_id_iterator<int>(offsets.begin(), offsets.end());
 * auto ids = std::vector<int>(10);
 * std::copy_n(iter_first, 10, ids.begin());  // ids now contains [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
 * @endcode
 *
 * @tparam GeometryIter The offset iterator type. Must meet the requirements
 * of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @param offsets_begin Beginning of range of geometry offsets
 * @param offsets_end End of range of geometry offsets
 * @return An iterator over unique IDs for each element of each geometry defined by the offsets
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename IndexT, typename GeometryIter>
auto make_geometry_id_iterator(GeometryIter offsets_begin, GeometryIter offsets_end)
{
  return detail::make_counting_transform_iterator(
    IndexT{0}, detail::index_to_geometry_id<IndexT, GeometryIter>{offsets_begin, offsets_end});
}

/**
 * @brief Create an input iterator that generates zero-based sequential geometry IDs for each
 * element of a nested geometry based on the input geometry and part offset ranges.
 *
 * This can be used for any two-level nested multigeometry offsets, e.g. multipolygons.
 *
 * Example:
 * @code
 * auto poly_offsets = std::vector<int>({0, 1, 3});  // poly 1 has 2 rings
 * auto ring_offsets = std::vector<int>({0, 4, 7, 10});
 * auto iter =
 *   make_geometry_id_iterator<int>(poly_offsets.begin(), poly_offsets.end(), ring_offsets.begin());
 * auto ids = std::vector<int>(13);
 * std::copy_n(iter, 13, ids.begin());
 * // ids now contains [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
 * @endcode
 *
 * @tparam GeometryIter The offset iterator type. Must meet the requirements
 * of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @param offsets_begin Beginning of range of geometry offsets
 * @param offsets_end End of range of geometry offsets
 * @param part_offsets_begin Beginning of range of part (e.g. ring) offsets
 *
 * @return An iterator over unique IDs for each element of each geometry defined by the offsets
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename IndexT, typename GeometryIter, typename PartIter>
auto make_geometry_id_iterator(GeometryIter geometry_offsets_begin,
                               GeometryIter geometry_offsets_end,
                               PartIter part_offsets_begin)
{
  auto first_part_offsets_begin =
    thrust::make_permutation_iterator(part_offsets_begin, geometry_offsets_begin);

  return make_geometry_id_iterator<IndexT>(
    first_part_offsets_begin,
    thrust::next(first_part_offsets_begin,
                 std::distance(geometry_offsets_begin, geometry_offsets_end)));
}

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
