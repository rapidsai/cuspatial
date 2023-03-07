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
#include <cuspatial/detail/utility/offset_to_keys.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/experimental/ranges/range.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>

#include <cstdint>
#include <type_traits>

namespace cuspatial {
namespace detail {

template <typename MultiPointRange, typename MultiPolygonRange>
struct point_in_multipolygon_test_functor {
  MultiPointRange multipoints;
  MultiPolygonRange multipolygons;

  point_in_multipolygon_test_functor(MultiPointRange multipoints, MultiPolygonRange multipolygons)
    : multipoints(multipoints), multipolygons(multipolygons)
  {
  }

  template <typename IndexType>
  uint8_t __device__ operator()(IndexType pidx)
  {
    printf("%d\n", static_cast<int>(pidx));

    auto point = thrust::raw_reference_cast(multipoints.point(pidx));

    printf("%f, %f\n", point.x, point.y);

    auto geometry_idx = multipoints.geometry_idx_from_point_idx(pidx);

    printf("%d\n", static_cast<int>(geometry_idx));

    bool intersects = false;
    for (auto polygon : multipolygons[geometry_idx]) {
      printf("here\n");
      intersects = intersects || is_point_in_polygon(point, polygon);
    }

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

  T dist_squared = std::numeric_limits<T>::max();
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multipolygons.num_points();
       idx += gridDim.x * blockDim.x) {
    auto geometry_idx = multipolygons.geometry_idx_from_segment_idx(idx);
    if (geometry_idx == MultiPolygonRange::INVALID_INDEX) continue;

    if (intersects[geometry_idx]) {
      // TODO: only the leading thread of the pair need to store the result, atomics is not needed.
      atomicMin(&distances[geometry_idx], T{0.0});
      continue;
    }

    printf("In distance kernel: %d %d %d",
           static_cast<int>(idx),
           static_cast<int>(geometry_idx),
           static_cast<int>(intersects.size()));

    auto [a, b] = multipolygons.get_segment(idx);
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
  using T = typename MultiPointRange::element_t;

  static_assert(is_same_floating_point<T, typename MultiPolygonRange::element_t>(),
                "Inputs must have same floating point value type.");

  static_assert(
    is_same<vec_2d<T>, typename MultiPointRange::point_t, typename MultiPolygonRange::point_t>(),
    "Inputs must be cuspatial::vec_2d");

  CUSPATIAL_EXPECTS(multipoints.size() == multipolygons.size(),
                    "Must have the same number of input rows.");

  auto multipoint_intersects = [&]() {
    rmm::device_uvector<uint8_t> point_intersects(multipoints.num_points(), stream);

    thrust::tabulate(rmm::exec_policy(stream),
                     point_intersects.begin(),
                     point_intersects.end(),
                     detail::point_in_multipolygon_test_functor{multipoints, multipolygons});

    rmm::device_uvector<uint8_t> multipoint_intersects(multipoints.num_multipoints(), stream);
    auto offset_as_key_it = detail::make_counting_transform_iterator(
      0, offsets_to_keys_functor{multipoints.offsets_begin(), multipoints.offsets_end()});

    thrust::reduce_by_key(rmm::exec_policy(stream),
                          offset_as_key_it,
                          offset_as_key_it + multipoints.num_points(),
                          point_intersects.begin(),
                          thrust::make_discard_iterator(),
                          multipoint_intersects.begin(),
                          thrust::logical_or<uint8_t>());

    return multipoint_intersects;
  }();

  auto [threads_per_block, n_blocks] = grid_1d(multipolygons.num_points());

  detail::
    pairwise_point_polygon_distance_kernel<<<threads_per_block, n_blocks, 0, stream.value()>>>(
      multipoints, multipolygons, multipoint_intersects.begin(), distances_first);

  CUSPATIAL_CHECK_CUDA(stream.value());

  return distances_first + multipoints.size();
}

}  // namespace cuspatial
