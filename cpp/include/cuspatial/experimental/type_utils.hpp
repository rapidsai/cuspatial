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

#include <cuspatial/types.hpp>
#include <cuspatial/utility/vec_2d.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <type_traits>

namespace cuspatial {

namespace detail {

/**
 * @internal
 * @brief Helper to convert a tuple of elements into a `vec_2d`
 */
template <typename T, typename VectorType>
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
template <typename T, typename VectorType>
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
 * @tparam VectorType cuSpatial vector type, must be `vec_2d`, `lonlat_2d` or `cartesian_2d`
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
template <typename VectorType, typename FirstIter, typename SecondIter>
auto make_vec_2d_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  static_assert(std::is_same_v<T, typename std::iterator_traits<SecondIter>::value_type>,
                "Iterator value_type mismatch");

  auto zipped = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::make_transform_iterator(zipped, detail::tuple_to_vec_2d<T, VectorType>());
}

/**
 * @brief Create an iterator to `lonlat_2d` data from two input iterators.
 *
 * Interleaves longitude and latitude from separate iterators into a single iterator to lon/lat
 * coordinates.
 * @tparam FirstIter Iterator type to the first component (the longitude) of `lonlat_2d`. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam SecondIter Iterator type to the second component (the latitude) of `lonlat_2d`. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @param first Iterator to beginning of `lonlat_2d::x`
 * @param second Iterator to beginning of `lonlat_2d::y`
 * @return Iterator to `lonlat_2d`
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
  return make_vec_2d_iterator<vec_2d<T>>(first, second);
}

template <typename FirstIter, typename SecondIter>
auto make_lonlat_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  return make_vec_2d_iterator<lonlat_2d<T>>(first, second);
}

/**
 * @brief Create an iterator to `cartesian_2d` data from two input iterators.
 *
 * Interleaves x and y coordinates from separate iterators into a single iterator to x-y
 * coordinates.
 * @tparam FirstIter Iterator type to the first component of `cartesian_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam SecondIter Iterator type to the second component of `cartesian_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @param first Iterator to beginning of `cartesian_2d::x`
 * @param second Iterator to beginning of `cartesian_2d::y`
 * @return Iterator to `cartesian_2d`
 *
 * @pre `first` and `second` must iterate on same data type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename FirstIter, typename SecondIter>
auto make_cartesian_2d_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  return make_vec_2d_iterator<cartesian_2d<T>>(first, second);
}

/**
 * @brief Create an output iterator to `vec_2d` data from two output iterators.
 *
 * Creates an output iterator from separate iterators to x and y data to which
 * can be written interleaved x/y data. This allows using two separate arrays of
 * output data with APIs that expect an iterator to structured data.
 *
 * @tparam VectorType cuSpatial vector type, must be `vec_2d`, `lonlat_2d` or `cartesian_2d`
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
template <typename VectorType, typename FirstIter, typename SecondIter>
auto make_zipped_vec_2d_output_iterator(FirstIter first, SecondIter second)
{
  using T         = typename std::iterator_traits<FirstIter>::value_type;
  auto zipped_out = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::make_transform_output_iterator(zipped_out,
                                                detail::vec_2d_to_tuple<T, VectorType>());
}

/**
 * @brief Create an output iterator to `lonlat_2d` from two output iterators.
 *
 * Creates an output iterator from separate iterators to longitude and latitude data
 * to which can be written interleaved longitude/latitude data. This allows using two
 * separate arrays of output data with APIs that expect an iterator to interleaved
 * data.
 *
 * @tparam FirstIter Iterator type to the first component of `lonlat_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @tparam SecondIter Iterator type to the second component of `lonlat_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @param first Iterator to beginning of longitude data.
 * @param second Iterator of beginning of latitude data.
 * @return Iterator to `lonlat_2d`
 *
 * @pre `first` and `second` must iterate on same data type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename FirstIter, typename SecondIter>
auto make_zipped_lonlat_output_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  return make_zipped_vec_2d_output_iterator<lonlat_2d<T>>(first, second);
}

/**
 * @brief Create an output iterator to `cartesian_2d` from two output iterators.
 *
 * Creates an output iterator from separate iterators to x and y data to which
 * can be written interleaved x/y data. This allows using two separate arrays of
 * output data with APIs that expect an iterator to structured data.
 * @tparam FirstIter Iterator type to the first component of `cartesian_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @tparam SecondIter Iterator type to the second component of `cartesian_2d`. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 * @param first Iterator to beginning of `x` data.
 * @param second Iterator to beginning of `y` data.
 * @return Iterator to `cartesian_2d`
 *
 * @pre `first` and `second` must iterate on same data type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename FirstIter, typename SecondIter>
auto make_zipped_cartesian_2d_output_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  return make_zipped_vec_2d_output_iterator<cartesian_2d<T>>(first, second);
}

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
