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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

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

}  // namespace detail

/**
 * @addtogroup type_factories
 * @{
 */

/**
 * @brief Create an iterator to `vec_2d` data from two input iterators.
 *
 * Interleaves x and y coordinates from separate iterators into a single iterator to x-y
 * coordinates.
 *
 * @tparam VectorType cuSpatial vector type, must be `vec_2d`
 * @tparam FirstIter Iterator type to the first component of `vec_2d`. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam SecondIter Iterator type to the second component of `vec_2d`. Must meet the requirements
 * of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
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

  auto zipped = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::make_transform_iterator(zipped, detail::tuple_to_vec_2d<T>());
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
auto make_zipped_vec_2d_output_iterator(FirstIter first, SecondIter second)
{
  using T         = typename std::iterator_traits<FirstIter>::value_type;
  auto zipped_out = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::make_transform_output_iterator(zipped_out, detail::vec_2d_to_tuple<T>());
}

template <typename T, typename VectorType = vec_2d<T>>
struct interleaved_to_vec_2d {
  T const* it;
  __device__ VectorType operator()(std::size_t i) { return VectorType{it[2 * i], it[2 * i + 1]}; }
};

template <typename Iter>
auto interleaved_iterator_to_vec_2d_iterator(Iter d_points_begin)
{
  using T = typename std::iterator_traits<Iter>::value_type;
  return detail::make_counting_transform_iterator(0, interleaved_to_vec_2d<T>{d_points_begin});
}

struct strided_functor {
  std::size_t _stride;
  strided_functor(std::size_t stride) : _stride(stride) {}
  auto __device__ operator()(std::size_t i) { return i * _stride; }
};

/**
 * @brief Create an output iterator to `vec_2d` data from an iterator to an interleaved array.
 *
 * @tparam Iter type of iterator to interleaved data. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @param d_points_begin Iterator to beginning of interleaved data.
 * @return Iterator to `vec_2d`
 */
template <typename Iter>
auto vec_2d_iterator_to_output_interleaved_iterator(Iter d_points_begin)
{
  using T                     = typename std::iterator_traits<Iter>::value_type;
  auto fixed_stride_2_functor = strided_functor(2);
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
 * @} // end of doxygen group
 */

}  // namespace cuspatial
