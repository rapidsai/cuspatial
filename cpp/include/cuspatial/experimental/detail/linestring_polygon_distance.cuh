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

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/vec_2d.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cuspatial {
namespace detail {

/**
 * @brief For each point in the multipoint, compute point-in-multipolygon in corresponding pair.
 */
template <typename MultiPointRange, typename MultiPolygonRange>
struct point_in_multipolygon_test_functor {
  using T = typename MultiPointRange::element_t;

  MultiPointRange multipoints;
  MultiPolygonRange multipolygons;

  point_in_multipolygon_test_functor(MultiPointRange multipoints, MultiPolygonRange multipolygons)
    : multipoints(multipoints), multipolygons(multipolygons)
  {
  }

  template <typename IndexType>
  uint8_t __device__ operator()(IndexType pidx)
  {
    vec_2d<T> const& point = multipoints.point(pidx);
    auto geometry_idx      = multipoints.geometry_idx_from_point_idx(pidx);

    auto const& polys = multipolygons[geometry_idx];
    // TODO: benchmark against range based for loop
    bool intersects =
      thrust::any_of(thrust::seq, polys.begin(), polys.end(), [&point] __device__(auto poly) {
        return is_point_in_polygon(point, poly);
      });

    return static_cast<uint8_t>(intersects);
  }
};

}  // namespace detail

template <class MultiLinestringRange, class MultiPolygonRange, class OutputIt>
OutputIt pairwise_linestring_polygon_distance(MultiLinestringRange multilinestrings,
                                              MultiPolygonRange multipolygons,
                                              OutputIt distances_first,
                                              rmm::cuda_stream_view stream)
{
  using T       = typename MultiLinestringRange::element_t;
  using index_t = typename MultiLinestringRange::index_t;

  CUSPATIAL_EXPECTS(multilinestrings.size() == multipolygons.size(),
                    "Must have the same number of input rows.");

  if (multilinestrings.size() == 0) return distances_first;

  // Create a multipoint range from multilinestrings
  auto multipoints = multilinestrings.as_multipoint_range();

  // Compute whether each multipoint intersects with the corresponding multipolygon.
  // First, compute the point-multipolygon intersection. Then use reduce-by-key to
  // compute the multipoint-multipolygon intersection.
  auto multipoint_intersects = [&]() {
    rmm::device_uvector<uint8_t> point_intersects(multipoints.num_points(), stream);

    thrust::tabulate(rmm::exec_policy(stream),
                     point_intersects.begin(),
                     point_intersects.end(),
                     detail::point_in_multipolygon_test_functor{multipoints, multipolygons});

    // `multipoints` contains only single points, no need to reduce.
    if (multipoints.is_single_point_range()) return point_intersects;

    rmm::device_uvector<uint8_t> multipoint_intersects(multipoints.num_multipoints(), stream);
    detail::zero_data_async(multipoint_intersects.begin(), multipoint_intersects.end(), stream);

    auto offset_as_key_it =
      make_geometry_id_iterator<index_t>(multipoints.offsets_begin(), multipoints.offsets_end());

    thrust::reduce_by_key(rmm::exec_policy(stream),
                          offset_as_key_it,
                          offset_as_key_it + multipoints.num_points(),
                          point_intersects.begin(),
                          thrust::make_discard_iterator(),
                          multipoint_intersects.begin(),
                          thrust::logical_or<uint8_t>());

    return multipoint_intersects;
  }();
}

}  // namespace cuspatial
