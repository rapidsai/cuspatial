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

#include <type_traits>

namespace cuspatial {

namespace detail {

template <typename T>
struct tuple_to_vec_2d {
  __device__ cuspatial::vec_2d<T> operator()(thrust::tuple<T, T> pos)
  {
    return cuspatial::vec_2d<T>{thrust::get<0>(pos), thrust::get<1>(pos)};
  }
};

template <typename T>
struct vec_2d_to_tuple {
  __device__ thrust::tuple<T, T> operator()(cuspatial::vec_2d<T> xy)
  {
    return thrust::make_tuple(xy.x, xy.y);
  }
};

}  // namespace detail

template <typename FirstIter, typename SecondIter>
auto make_vec_2d_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  static_assert(std::is_same_v<T, typename std::iterator_traits<SecondIter>::value_type>,
                "Iterator value_type mismatch");

  auto zipped = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::make_transform_iterator(zipped, detail::tuple_to_vec_2d<T>());
}

template <typename FirstIter, typename SecondIter>
auto make_zipped_vec_2d_output_iterator(FirstIter first, SecondIter second)
{
  using T         = typename std::iterator_traits<FirstIter>::value_type;
  auto zipped_out = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::make_transform_output_iterator(zipped_out, detail::vec_2d_to_tuple<T>());
}

}  // namespace cuspatial
