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
struct tuple_to_location_2d {
  __device__ cuspatial::location_2d<T> operator()(thrust::tuple<T, T> lonlat)
  {
    return cuspatial::location_2d<T>{thrust::get<0>(lonlat), thrust::get<1>(lonlat)};
  }
};

template <typename T>
struct tuple_to_coordinate_2d {
  __device__ cuspatial::coordinate_2d<T> operator()(thrust::tuple<T, T> xy)
  {
    return cuspatial::coordinate_2d<T>{thrust::get<0>(xy), thrust::get<1>(xy)};
  }
};

template <typename T>
struct coordinate_2d_to_tuple {
  __device__ thrust::tuple<T, T> operator()(cuspatial::coordinate_2d<T> xy)
  {
    return thrust::make_tuple(xy.x, xy.y);
  }
};

}  // namespace detail

// convert two iterators to an iterator<location_2d<T>>
template <typename LonIter, typename LatIter>
auto to_location_2d(LonIter lon, LatIter lat)
{
  using T = typename std::iterator_traits<LonIter>::value_type;
  static_assert(std::is_same_v<T, typename std::iterator_traits<LatIter>::value_type>,
                "Iterator value_type mismatch");

  auto zipped = thrust::make_zip_iterator(thrust::make_tuple(lon, lat));
  return thrust::make_transform_iterator(zipped, detail::tuple_to_location_2d<T>());
}

// convert two iterators to an iterator<coordinate_2d<T>>
template <typename XIter, typename YIter>
auto to_coordinate_2d(XIter lon, YIter lat)
{
  using T = typename std::iterator_traits<XIter>::value_type;
  static_assert(std::is_same_v<T, typename std::iterator_traits<YIter>::value_type>,
                "Iterator value_type mismatch");

  auto zipped = thrust::make_zip_iterator(thrust::make_tuple(lon, lat));
  return thrust::make_transform_iterator(zipped, detail::tuple_to_coordinate_2d<T>());
}

template <typename T, typename OutputIter>
auto coordinate_2d_to_zip(OutputIter out)
{
  return thrust::make_transform_output_iterator(out, detail::coordinate_2d_to_tuple<T>());
}

}  // namespace cuspatial
