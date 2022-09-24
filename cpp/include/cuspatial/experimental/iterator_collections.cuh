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

#include "cuspatial/cuda_utils.hpp"
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/detail/utility/traits.hpp>
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

  CUSPATIAL_HOST_DEVICE
  auto begin() const { return point_begin(); }
  CUSPATIAL_HOST_DEVICE
  auto end() const { return point_end(); }

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

  auto size() { return thrust::distance(geometry_begin, geometry_end) - 1; }

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

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto element(IndexType idx)
  {
    return multipoint{points_begin + geometry_begin[idx], points_end + geometry_begin[idx + 1]};
  }

 protected:
  GeometryIterator geometry_begin;
  GeometryIterator geometry_end;
  VecIterator points_begin;
  VecIterator points_end;
};

template <typename GeometryIterator, typename PartIterator, typename VecIterator>
class multilinestring_array {
  using T = cuspatial::detail::iterator_vec_base_type<VecIterator>;

 public:
  multilinestring_array(GeometryIterator geometry_begin,
                        GeometryIterator geometry_end,
                        PartIterator part_begin,
                        PartIterator part_end,
                        VecIterator points_begin,
                        VecIterator points_end)
    : geometry_begin(geometry_begin),
      geometry_end(geometry_end),
      part_begin(part_begin),
      part_end(part_end),
      points_begin(points_begin),
      points_end(points_end)
  {
  }

  CUSPATIAL_HOST_DEVICE auto size() { return num_multilinestrings(); }

  CUSPATIAL_HOST_DEVICE
  auto num_multilinestrings() { return thrust::distance(geometry_begin, geometry_end) - 1; }
  CUSPATIAL_HOST_DEVICE auto num_linestrings()
  {
    return thrust::distance(part_begin, part_end) - 1;
  }

  CUSPATIAL_HOST_DEVICE auto num_points() { return thrust::distance(points_begin, points_end) - 1; }
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto part_idx_from_point_idx(IndexType point_idx)
  {
    auto part_it = thrust::upper_bound(thrust::seq, part_begin, part_end, point_idx);
    return thrust::distance(part_begin, thrust::prev(part_it));
  }

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_part_idx(IndexType part_idx)
  {
    auto geom_it = thrust::upper_bound(thrust::seq, geometry_begin, geometry_end, part_idx);
    return thrust::distance(geometry_begin, thrust::prev(geom_it));
  }

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_point_idx(IndexType point_idx)
  {
    return geometry_idx_from_part_idx(part_idx_from_point_idx(point_idx));
  }

  /**
   * @internal
   * A segment id is the same as the id to the starting point of the segment.
   * A segment id is valid iff the id is not the last point in the linestring.
   */
  template <typename IndexType1, typename IndexType2>
  CUSPATIAL_HOST_DEVICE bool is_valid_segment_id(IndexType1 segment_idx, IndexType2 part_idx)
  {
    return segment_idx != num_points() && segment_idx < part_begin[part_idx + 1];
  }

  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE thrust::pair<vec_2d<T>, vec_2d<T>> segment(IndexType segment_idx)
  {
    return thrust::make_pair(points_begin[segment_idx], points_begin[segment_idx + 1]);
  }

 protected:
  GeometryIterator geometry_begin;
  GeometryIterator geometry_end;
  PartIterator part_begin;
  PartIterator part_end;
  VecIterator points_begin;
  VecIterator points_end;
};

}  // namespace iterator_collections
}  // namespace cuspatial
