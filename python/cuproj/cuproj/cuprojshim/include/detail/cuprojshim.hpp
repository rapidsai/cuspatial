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

#include <cuproj/projection.cuh>
#include <cuproj/projection_factories.cuh>
#include <cuproj/vec_2d.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <iterator>
#include <type_traits>

namespace cuprojshim {

namespace detail {
/**
 * @internal
 * @brief Helper to convert a tuple of elements into a `vec_2d`
 */
template <typename T, typename VectorType = cuproj::vec_2d<T>>
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
template <typename T, typename VectorType = cuproj::vec_2d<T>>
struct vec_2d_to_tuple {
  __device__ thrust::tuple<T, T> operator()(VectorType const& xy)
  {
    return thrust::make_tuple(xy.x, xy.y);
  }
};

/**
 * @brief Create an iterator to `vec_2d` data from two input iterators.
 *
 * Interleaves x and y coordinates from separate iterators into a single
 * iterator to xy- coordinates.
 *
 * @tparam VectorType cuSpatial vector type, must be `vec_2d`
 * @tparam FirstIter Iterator type to the first component of `vec_2d`. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-accessible.
 * @tparam SecondIter Iterator type to the second component of `vec_2d`. Must
 * meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-accessible.
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
  static_assert(std::is_same<T, typename std::iterator_traits<SecondIter>::value_type>(),
                "Iterator value_type mismatch");

  auto zipped = thrust::make_zip_iterator(first, second);
  return thrust::make_transform_iterator(zipped, tuple_to_vec_2d<T>());
}

/**
 * @brief Create an output iterator to `vec_2d` data from two output iterators.
 *
 * Creates an output iterator from separate iterators to x and y data to which
 * can be written interleaved x/y data. This allows using two separate arrays of
 * output data with APIs that expect an iterator to structured data.
 *
 * @tparam VectorType cuSpatial vector type, must be `vec_2d`
 * @tparam FirstIter Iterator type to the first component of `vec_2d`. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be
 * device-accessible.
 * @tparam SecondIter Iterator type to the second component of `vec_2d`. Must
 * meet the requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable
 * and be device-accessible.
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
  return thrust::transform_output_iterator(zipped_out, vec_2d_to_tuple<T>());
}
}  // namespace detail

template <typename T>
cuproj::projection<cuproj::vec_2d<T>>* make_projection(std::string const& src_epsg,
                                                       std::string const& dst_epsg)
{
  return cuproj::make_projection<cuproj::vec_2d<T>>(src_epsg, dst_epsg);
}

template <typename T>
void transform(cuproj::projection<cuproj::vec_2d<T>> const& proj,
               cuproj::vec_2d<T>* xy_in,
               cuproj::vec_2d<T>* xy_out,
               std::size_t n,
               cuproj::direction dir)
{
  proj.transform(xy_in, xy_in + n, xy_out, dir);
}

template <typename T>
void transform(cuproj::projection<cuproj::vec_2d<T>> const& proj,
               T* x_in,
               T* y_in,
               T* x_out,
               T* y_out,
               std::size_t n,
               cuproj::direction dir)
{
  auto xy_in  = detail::make_vec_2d_iterator(x_in, y_in);
  auto xy_out = detail::make_vec_2d_output_iterator(x_out, y_out);
  proj.transform(xy_in, xy_in + n, xy_out, dir);
}

}  // namespace cuprojshim
