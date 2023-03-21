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
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>

#include <cstdint>
#include <limits>
#include <type_traits>

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

/**
 * @brief Kernel to compute the distance between pairs of point and polygon.
 */
template <class MultiPointRange,
          class MultiPolygonRange,
          class IntersectionRange,
          class OutputIterator>
void __global__ pairwise_point_polygon_distance_kernel(MultiPointRange multipoints,
                                                       MultiPolygonRange multipolygons,
                                                       IntersectionRange intersects,
                                                       OutputIterator distances)
{
  using T = typename MultiPointRange::element_t;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multipolygons.num_points();
       idx += gridDim.x * blockDim.x) {
    auto geometry_idx = multipolygons.geometry_idx_from_segment_idx(idx);
    if (geometry_idx == MultiPolygonRange::INVALID_INDEX) continue;

    if (intersects[geometry_idx]) {
      distances[geometry_idx] = T{0.0};
      continue;
    }

    auto [a, b] = multipolygons.get_segment(idx);

    T dist_squared = std::numeric_limits<T>::max();
    for (vec_2d<T> point : multipoints[geometry_idx]) {
      dist_squared = min(dist_squared, point_to_segment_distance_squared(point, a, b));
    }

    atomicMin(&distances[geometry_idx], sqrt(dist_squared));
  }
}

}  // namespace detail

template <class MultiPointRange, class MultiPolygonRange, class OutputIt>
OutputIt pairwise_point_polygon_distance(MultiPointRange multipoints,
                                         MultiPolygonRange multipolygons,
                                         OutputIt distances_first,
                                         rmm::cuda_stream_view stream)
{
  using T       = typename MultiPointRange::element_t;
  using index_t = typename MultiPointRange::index_t;

  CUSPATIAL_EXPECTS(multipoints.size() == multipolygons.size(),
                    "Must have the same number of input rows.");

  if (multipoints.size() == 0) return distances_first;

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

  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + multipoints.size(),
               std::numeric_limits<T>::max());
  auto [threads_per_block, n_blocks] = grid_1d(multipolygons.num_points());

  detail::
    pairwise_point_polygon_distance_kernel<<<n_blocks, threads_per_block, 0, stream.value()>>>(
      multipoints, multipolygons, multipoint_intersects.begin(), distances_first);

  CUSPATIAL_CHECK_CUDA(stream.value());

  return distances_first + multipoints.size();
}

}  // namespace cuspatial
