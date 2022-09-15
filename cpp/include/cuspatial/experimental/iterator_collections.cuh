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

#include <thrust/iterator/transform_iterator.h>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/vec_2d.hpp>

#include <iterator>

namespace cuspatial {

namespace iterator_collections {

template <typename VecIterator>
class multipoint {
 public:
  CUSPATIAL_HOST_DEVICE
  multipoint(VecIterator begin, VecIterator end) : points_begin(begin), points_end(end) {}

  CUSPATIAL_HOST_DEVICE
  auto point_begin() const { return points_begin; }
  CUSPATIAL_HOST_DEVICE
  auto point_end() const { return points_end; }

 protected:
  VecIterator points_begin;
  VecIterator points_end;
};

template <typename GeometryIterator,
          typename VecIterator,
          typename T = typename std::iterator_traits<VecIterator>::value_type>
class multipoint_array {
 public:
  multipoint_array(GeometryIterator geom_begin,
                   GeometryIterator geom_end,
                   VecIterator point_begin,
                   VecIterator point_end)
    : geometry_begin(geom_begin),
      geometry_end(geom_end),
      points_begin(point_begin),
      points_end(point_end)
  {
  }

  struct to_multipoint_functor {
    using difference_type = typename thrust::iterator_difference<GeometryIterator>::type;
    GeometryIterator geometry_begin;
    VecIterator points_begin;

    to_multipoint_functor(GeometryIterator g, VecIterator p) : geometry_begin(g), points_begin(p) {}

    CUSPATIAL_HOST_DEVICE
    auto operator()(difference_type const& i)
    {
      return multipoint<VecIterator>{points_begin + geometry_begin[i],
                                     points_begin + geometry_begin[i + 1]};
    }
  };

  struct to_point_functor {
    using difference_type = typename thrust::iterator_difference<VecIterator>::type;
    VecIterator points_begin;

    to_point_functor(VecIterator p) : points_begin(p) {}

    CUSPATIAL_HOST_DEVICE
    vec_2d<T> operator()(difference_type const& i) { return points_begin[i]; }
  };

  auto multipoint_begin()
  {
    return detail::make_counting_transform_iterator(
      0, to_multipoint_functor(geometry_begin, points_begin));
  }

  auto multipoint_end()
  {
    return multipoint_begin() + thrust::distance(geometry_begin, geometry_end);
  }

  auto point_begin()
  {
    return detail::make_counting_transform_iterator(0, to_point_functor(points_begin));
  }

  auto point_end() { return point_begin() + thrust::distance(points_begin, points_end); }

 protected:
  GeometryIterator geometry_begin;
  GeometryIterator geometry_end;
  VecIterator points_begin;
  VecIterator points_end;
};
}  // namespace iterator_collections
}  // namespace cuspatial
