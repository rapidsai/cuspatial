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

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cuspatial {

namespace detail {
template <typename T, typename VectorType>
struct tuple_to_vec_2d {
  __device__ VectorType operator()(thrust::tuple<T, T> const& pos)
  {
    return VectorType{thrust::get<0>(pos), thrust::get<1>(pos)};
  }
};

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
 *
 * CuSpatial functions inside `experimental` folder are header-only and only accepts
 * input/output iterators on coordinates. These factory functions are convenient ways
 * to create iterators from data in various format.
 *
 * @{
 * @file
 * @brief Factory method to create coordinate iterators
 *
 * @copydetails type_factories
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

template <typename FirstIter, typename SecondIter>
auto make_lonlat_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  return make_vec_2d_iterator<lonlat_2d<T>>(first, second);
}

template <typename FirstIter, typename SecondIter>
auto make_cartesian_2d_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  return make_vec_2d_iterator<cartesian_2d<T>>(first, second);
}

template <typename VectorType, typename FirstIter, typename SecondIter>
auto make_zipped_vec_2d_output_iterator(FirstIter first, SecondIter second)
{
  using T         = typename std::iterator_traits<FirstIter>::value_type;
  auto zipped_out = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::make_transform_output_iterator(zipped_out,
                                                detail::vec_2d_to_tuple<T, VectorType>());
}

template <typename FirstIter, typename SecondIter>
auto make_zipped_lonlat_output_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  return make_zipped_vec_2d_output_iterator<lonlat_2d<T>>(first, second);
}

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
