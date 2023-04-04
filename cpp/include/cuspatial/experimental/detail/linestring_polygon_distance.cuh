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

#include "distance_utils.cuh"

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/experimental/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/experimental/detail/functors.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <libcudf/rapids/thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/zip_function.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>

namespace cuspatial {
namespace detail {

/**
 * @brief Computes distances between the multilinestring and multipolygons
 *
 * @param multilinestrings Range to the multilinestring
 * @param multipolygons Range to the multipolygon
 * @param thread_bounds Range to the boundary of thread partitions
 * @param multilinestrings_segment_offsets Range to the indices where the first segment of each
 * multilinestring begins
 * @param multipolygons_segment_offsets Range to the indices where the first segment of each
 * multipolygon begins
 * @param intersects A uint8_t array that indicates if the corresponding pair of multipoint and
 * multipolygon intersects
 * @param distances Output range of distances, pre-filled with std::numerical_limits<T>::max()
 */
template <typename MultiLinestringRange,
          typename MultiPolygonRange,
          typename IndexRange,
          typename OutputIt>
void __global__
pairwise_linestring_polygon_distance_kernel(MultiLinestringRange multilinestrings,
                                            MultiPolygonRange multipolygons,
                                            IndexRange thread_bounds,
                                            IndexRange multilinestrings_segment_offsets,
                                            IndexRange multipolygons_segment_offsets,
                                            uint8_t* intersects,
                                            OutputIt* distances)
{
  using T       = typename MultiLinestringRange::element_t;
  using index_t = iterator_value_type<typename MultiLinestringRange::geometry_it_t>;

  auto num_threads = thread_bounds[thread_bounds.size() - 1];
  for (auto idx = blockDim.x * blockIdx.x + threadIdx.x; idx < num_threads;
       idx += blockDim.x * gridDim.x) {
    auto it = thrust::prev(
      thrust::upper_bound(thrust::seq, thread_bounds.begin(), thread_bounds.end(), idx));
    auto geometry_id = thrust::distance(thread_bounds.begin(), it);
    auto local_idx   = idx - *it;

    if (intersects[geometry_id]) {
      distances[geometry_id] = 0.0f;
      continue;
    }

    // Retrieve the number of segments in multilinestrings[geometry_id]
    auto num_segment_this_multilinestring =
      multilinestrings.multilinestring_segment_count_begin()[geometry_id];
    // The segment id from the multilinestring this thread is compmuting (local_id + global_offset)
    auto multilinestring_segment_id =
      local_idx % num_segment_this_multilinestring + multilinestrings_segment_offsets[geometry_id];
    // The segment id from the multipolygon this thread is computing (local_id + global_offset)
    auto multipolygon_segment_id =
      local_idx / num_segment_this_multilinestring + multipolygons_segment_offsets[geometry_id];

    auto [a, b] = multilinestrings.segment_begin()[multilinestring_segment_id];
    auto [c, d] = multipolygons.segment_begin()[multipolygon_segment_id];

    auto distance = sqrt(squared_segment_distance(a, b, c, d));

    atomicMin(&distances[geometry_id], sqrt(squared_segment_distance(a, b, c, d)));
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
  using index_t = iterator_value_type<typename MultiLinestringRange::geometry_it_t>;

  CUSPATIAL_EXPECTS(multilinestrings.size() == multipolygons.size(),
                    "Must have the same number of input rows.");

  if (multilinestrings.size() == 0) return distances_first;

  // Create a multipoint range from multilinestrings, computes intersection
  auto multipoints           = multilinestrings.as_multipoint_range();
  auto multipoint_intersects = point_polygon_intersects(multipoints, multipolygons, stream);

  // Compute the "boundary" of threads. Threads are partitioned based on the number of linestrings
  // times the number of polygons in a multipoint-multipolygon pair.
  auto segment_count_product_it = thrust::make_transform_iterator(
    thrust::make_zip_iterator(multilinestrings.multilinestring_segment_count_begin(),
                              multipolygons.multipolygon_segment_count_begin()),
    thrust::make_zip_function(thrust::multiplies<index_t>{}));

  // Computes the "thread boundary" of each pair. This array partitions the thread range by
  // geometries. E.g. threadIdx within [thread_bounds[i], thread_bounds[i+1]) computes distances of
  // the ith pair.
  auto thread_bounds = rmm::device_uvector<index_t>(multilinestrings.size() + 1, stream);
  detail::zero_data_async(thread_bounds.begin(), thread_bounds.end(), stream);

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         segment_count_product_it,
                         segment_count_product_it + thread_bounds.size() - 1,
                         thrust::next(thread_bounds.begin()));

  // Compute offsets to the first segment of each multilinestring and multipolygon
  auto multilinestring_segment_offsets =
    rmm::device_uvector<index_t>(multilinestrings.num_multilinestrings() + 1, stream);
  detail::zero_data_async(
    multilinestring_segment_offsets.begin(), multilinestring_segment_offsets.end(), stream);

  auto multipolygon_segment_offsets =
    rmm::device_uvector<index_t>(multipolygons.num_multipolygons() + 1, stream);
  detail::zero_data_async(
    multipolygon_segment_offsets.begin(), multipolygon_segment_offsets.end(), stream);

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         multilinestrings.multilinestring_segment_count_begin(),
                         multilinestrings.multilinestring_segment_count_begin() +
                           multilinestrings.num_multilinestrings(),
                         thrust::next(multilinestring_segment_offsets.begin()));

  thrust::inclusive_scan(
    rmm::exec_policy(stream),
    multipolygons.multipolygon_segment_count_begin(),
    multipolygons.multipolygon_segment_count_begin() + multipolygons.num_multipolygons(),
    thrust::next(multipolygon_segment_offsets.begin()));

  // Initialize output range
  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + multilinestrings.num_multilinestrings(),
               std::numeric_limits<T>::max());

  auto num_threads       = thread_bounds.back_element(stream);
  auto [tpb, num_blocks] = grid_1d(num_threads);

  detail::pairwise_linestring_polygon_distance_kernel<<<num_blocks, tpb, 0, stream.value()>>>(
    multilinestrings,
    multipolygons,
    range{thread_bounds.begin(), thread_bounds.end()},
    range{multilinestring_segment_offsets.begin(), multilinestring_segment_offsets.end()},
    range{multipolygon_segment_offsets.begin(), multipolygon_segment_offsets.end()},
    multipoint_intersects.begin(),
    distances_first);

  return distances_first + multilinestrings.num_multilinestrings();
}

}  // namespace cuspatial
