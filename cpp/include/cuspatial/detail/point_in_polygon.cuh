/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuspatial/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/detail/utility/validation.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/tabulate.h>

#include <iterator>
#include <type_traits>

namespace cuspatial {

template <class PointRange, class PolygonRange>
struct pip_functor {
  PointRange points;
  PolygonRange polygons;

  int32_t __device__ operator()(std::size_t i)
  {
    auto point       = points[i][0];
    int32_t hit_mask = 0;
    for (auto poly_idx = 0; poly_idx < polygons[i].num_polygons(); ++poly_idx)
      hit_mask |= (is_point_in_polygon(point, polygons[i][poly_idx]) << poly_idx);
    return hit_mask;
  }
};

template <class PointRange, class PolygonRange>
pip_functor(PointRange, PolygonRange) -> pip_functor<PointRange, PolygonRange>;

template <class PointRange, class PolygonRange, class OutputIt>
OutputIt point_in_polygon(PointRange points,
                          PolygonRange polygons,
                          OutputIt output,
                          rmm::cuda_stream_view stream)
{
  using T = typename PointRange::element_t;

  static_assert(points.contains_only_single_geometry() && polygons.contains_only_single_geometry(),
                "pairwise_point_in_polygon only supports single-point to single-polygon tests.");

  static_assert(is_same_floating_point<T, typename PolygonRange::element_t>(),
                "points and polygons must have the same coordinate type.");

  static_assert(std::is_same_v<iterator_value_type<OutputIt>, int32_t>,
                "OutputIt must point to 32 bit integer type.");

  CUSPATIAL_EXPECTS(points.size() == polygons.size(),
                    "Must pass in an equal number of (multi)points and (multi)polygons");

  CUSPATIAL_EXPECTS(polygons.size() <= std::numeric_limits<int32_t>::digits,
                    "Number of polygons cannot exceed 31");

  thrust::tabulate(
    rmm::exec_policy(stream), output, output + points.size(), pip_functor{points, polygons});

  return output + points.size();
}

template <class PointRange, class PolygonRange, class OutputIt>
OutputIt pairwise_point_in_polygon(PointRange points,
                                   PolygonRange polygons,
                                   OutputIt output,
                                   rmm::cuda_stream_view stream)
{
  using T = typename PointRange::element_t;

  static_assert(points.contains_only_single_geometry() && polygons.contains_only_single_geometry(),
                "pairwise_point_in_polygon only supports single-point to single-polygon tests.");

  static_assert(is_same_floating_point<T, typename PolygonRange::element_t>(),
                "points and polygons must have the same coordinate type.");

  static_assert(std::is_same_v<iterator_value_type<OutputIt>, int32_t>,
                "OutputIt must point to 32 bit integer type.");

  CUSPATIAL_EXPECTS(points.size() == polygons.size(),
                    "Must pass in an equal number of (multi)points and (multi)polygons");

  return thrust::transform(rmm::exec_policy(stream),
                           points.begin(),
                           points.end(),
                           polygons.begin(),
                           output,
                           [] __device__(auto multipoint, auto multipolygon) {
                             return is_point_in_polygon(static_cast<vec_2d<T>>(multipoint[0]),
                                                        multipolygon[0]);
                           });
}

}  // namespace cuspatial
