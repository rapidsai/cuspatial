/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cuspatial/geometry/vec_3d.hpp>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/tabulate.h>

#include <iterator>
#include <type_traits>

namespace cuspatial {

/**
 * @brief Computes point-in-polygon result of a single point to up to 32 polygons
 * Result is stored in an `int32_t` integer.
 */
template <class PointRange, class PolygonRange>
struct pip_functor {
  PointRange multipoints;
  PolygonRange multipolygons;

  int32_t __device__ operator()(std::size_t i)
  {
    using T          = typename PointRange::element_t;
    using PointType  = typename PointRange::point_t;
    int32_t hit_mask = 0;

    if constexpr (is_same<vec_2d<T>, PointType>()) {
      vec_2d<T> point = multipoints[i][0];
      for (auto poly_idx = 0; poly_idx < multipolygons.size(); ++poly_idx) {
        hit_mask |= (is_point_in_polygon(point, multipolygons[poly_idx][0]) << poly_idx);
      }
    } else {
      vec_3d<T> point = multipoints[i][0];
      for (auto poly_idx = 0; poly_idx < multipolygons.size(); ++poly_idx) {
        hit_mask |= (is_point_in_polygon_spherical(point, multipolygons[poly_idx][0]) << poly_idx);
      }
    }
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

  static_assert(is_same_floating_point<T, typename PolygonRange::element_t>(),
                "points and polygons must have the same coordinate type.");

  static_assert(std::is_same_v<iterator_value_type<OutputIt>, int32_t>,
                "OutputIt must point to 32 bit integer type.");

  CUSPATIAL_EXPECTS(points.num_multipoints() == points.num_points(),
                    "Point in polygon API only support single point - single polygon tests. "
                    "Multipoint input is not accepted.");

  CUSPATIAL_EXPECTS(polygons.num_multipolygons() == polygons.num_polygons(),
                    "Point in polygon API only support single point - single polygon tests. "
                    "MultiPolygon input is not accepted.");

  CUSPATIAL_EXPECTS(polygons.size() <= std::numeric_limits<int32_t>::digits,
                    "Number of polygons cannot exceed 31");

  if (points.size() == 0) return output;

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
  using T         = typename PointRange::element_t;
  using PointType = typename PointRange::point_t;

  static_assert(is_same_floating_point<T, typename PolygonRange::element_t>(),
                "points and polygons must have the same coordinate type.");

  static_assert(std::is_same_v<iterator_value_type<OutputIt>, uint8_t>,
                "OutputIt must be iterator to a uint8_t range.");

  CUSPATIAL_EXPECTS(points.num_multipoints() == points.num_points(),
                    "Point in polygon API only supports single point - single polygon tests. "
                    "Multipoint input is not accepted.");

  CUSPATIAL_EXPECTS(polygons.num_multipolygons() == polygons.num_polygons(),
                    "Point in polygon API only supports single point - single polygon tests. "
                    "MultiPolygon input is not accepted.");

  CUSPATIAL_EXPECTS(points.size() == polygons.size(),
                    "Must pass in an equal number of (multi)points and (multi)polygons");

  if (points.size() == 0) return output;

  return thrust::transform(rmm::exec_policy(stream),
                           points.begin(),
                           points.end(),
                           polygons.begin(),
                           output,
                           [] __device__(auto multipoint, auto multipolygon) {
                             if constexpr (is_same<vec_2d<T>, PointType>()) {
                               return is_point_in_polygon(static_cast<vec_2d<T>>(multipoint[0]),
                                                          multipolygon[0]);
                             } else {
                               return is_point_in_polygon_spherical(
                                 static_cast<vec_3d<T>>(multipoint[0]), multipolygon[0]);
                             }
                           });
}

}  // namespace cuspatial
