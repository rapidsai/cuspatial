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

template <typename CoordType, typename T = typename CoordType::value_type>
struct tuple_to_coord_2d {
  __device__ CoordType operator()(thrust::tuple<T, T> pos)
  {
    static_assert(std::is_base_of_v<coord_2d<T>, CoordType>(),
                  "Can only convert to coord_2d type.");
    return CoordType{thrust::get<0>(pos), thrust::get<1>(pos)};
  }
};

}  // namespace detail

template <typename CoordType, typename FirstIter, typename SecondIter>
auto make_coord_2d_iterator(FirstIter first, SecondIter second)
{
  using T = typename std::iterator_traits<FirstIter>::value_type;
  static_assert(std::is_same_v<T, typename std::iterator_traits<SecondIter>::value_type>,
                "Iterator value_type mismatch");

  auto zipped = thrust::make_zip_iterator(thrust::make_tuple(first, second));
  return thrust::make_transform_iterator(zipped, detail::tuple_to_coord_2d<CoordType>());
}

}  // namespace cuspatial
