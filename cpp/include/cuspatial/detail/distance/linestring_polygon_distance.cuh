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
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/range.cuh>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>

namespace cuspatial {
namespace detail {

/// Device functor that returns true if any of the geometry is empty.
struct any_input_is_empty {
  template <typename LhsType, typename RhsType>
  bool __device__ operator()(LhsType lhs, RhsType rhs)
  {
    return lhs.is_empty() || rhs.is_empty();
  }
};

/**
 * @brief Computes distances between the multilinestring and multipolygons
 *
 * This is a load balanced distance compute kernel. Each thread compute exactly 1 pair of segments
 * between the multilinestring and multipolygon.
 *
 * @tparam T type of the underlying coordinates
 * @tparam index_t type of underlying offsets
 *
 * @param multilinestring_segments Range to the segments of the multilinestring
 * @param multipolygon_segments Range to the segments of the multipolygon
 * @param thread_bounds Range to the boundary of thread partitions
 * @param intersects A uint8_t array that indicates if the corresponding pair of multipoint and
 * multipolygon intersects
 * @param distances Output range of distances, pre-filled with std::numerical_limits<T>::max()
 */
template <typename T,
          typename index_t,
          typename MultiLinestringSegmentRange,
          typename MultiPolygonSegmentRange,
          typename IndexRange,
          typename OutputIt>
void __global__
pairwise_linestring_polygon_distance_kernel(MultiLinestringSegmentRange multilinestring_segments,
                                            MultiPolygonSegmentRange multipolygon_segments,
                                            IndexRange thread_bounds,
                                            uint8_t* intersects,
                                            OutputIt* distances)
{
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
      multilinestring_segments.multigeometry_count_begin()[geometry_id];
    // The segment id from the multilinestring this thread is computing (local_id + global_offset)
    auto multilinestring_segment_id =
      local_idx % num_segment_this_multilinestring +
      multilinestring_segments.multigeometry_offset_begin()[geometry_id];
    // The segment id from the multipolygon this thread is computing (local_id + global_offset)
    auto multipolygon_segment_id = local_idx / num_segment_this_multilinestring +
                                   multipolygon_segments.multigeometry_offset_begin()[geometry_id];

    auto [a, b] = multilinestring_segments.begin()[multilinestring_segment_id];
    auto [c, d] = multipolygon_segments.begin()[multipolygon_segment_id];

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

  auto size = multilinestrings.size();

  if (size == 0) return distances_first;

  // Create a multipoint range from multilinestrings, computes intersection
  auto multipoints           = multilinestrings.as_multipoint_range();
  auto multipoint_intersects = point_polygon_intersects(multipoints, multipolygons, stream);

  // Make views to the segments in the multilinestring
  auto multilinestring_segments       = multilinestrings._segments(stream);
  auto multilinestring_segments_range = multilinestring_segments.view();
  auto multilinestring_segment_count_begin =
    multilinestring_segments_range.multigeometry_count_begin();

  // Make views to the segments in the multilinestring
  auto multipolygon_segments            = multipolygons._segments(stream);
  auto multipolygon_segments_range      = multipolygon_segments.view();
  auto multipolygon_segment_count_begin = multipolygon_segments_range.multigeometry_count_begin();

  // Compute the "boundary" of threads. Threads are partitioned based on the number of linestrings
  // times the number of polygons in a multilinestring-multipolygon pair.
  auto segment_count_product_it =
    thrust::make_transform_iterator(thrust::make_zip_iterator(multilinestring_segment_count_begin,
                                                              multipolygon_segment_count_begin),
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

  // Initialize output range
  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + size,
               std::numeric_limits<T>::max());

  // If any input multigeometries is empty, result is nan.
  auto nan_it = thrust::make_constant_iterator(std::numeric_limits<T>::quiet_NaN());
  thrust::transform_if(rmm::exec_policy(stream),
                       nan_it,
                       nan_it + size,
                       thrust::make_zip_iterator(multilinestrings.begin(), multipolygons.begin()),
                       distances_first,
                       thrust::identity<T>{},
                       thrust::make_zip_function(detail::any_input_is_empty{}));

  auto num_threads       = thread_bounds.back_element(stream);
  auto [tpb, num_blocks] = grid_1d(num_threads);

  detail::pairwise_linestring_polygon_distance_kernel<T, index_t>
    <<<num_blocks, tpb, 0, stream.value()>>>(multilinestring_segments_range,
                                             multipolygon_segments_range,
                                             range{thread_bounds.begin(), thread_bounds.end()},
                                             multipoint_intersects.begin(),
                                             distances_first);

  return distances_first + size;
}

}  // namespace cuspatial
