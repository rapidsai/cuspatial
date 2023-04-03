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

#include <cuspatial_test/test_util.cuh>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/experimental/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/experimental/detail/functors.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>

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

    int i           = 0;
    bool intersects = thrust::any_of(
      thrust::seq,
      polys.begin(),
      polys.end(),
      [&point, &i, &pidx, &geometry_idx] __device__(auto poly) {
        vec_2d<T> first_point = poly.point_begin()[0];
        printf(
          "pidx: %d, geometry_idx: %d, (%f, %f) in poly %d (first point: (%f %f), size: %d)?\n",
          static_cast<int>(pidx),
          static_cast<int>(geometry_idx),
          point.x,
          point.y,
          i,
          first_point.x,
          first_point.y,
          static_cast<int>(poly.size()));
        bool res = is_point_in_polygon(point, poly);
        printf("res: %d\n", res);
        ++i;
        return res;
      });

    return static_cast<uint8_t>(intersects);
  }
};

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

    printf("idx: %d, geometry_id: %d, intersects?: %d, local_idx: %d\n",
           static_cast<int>(idx),
           static_cast<int>(geometry_id),
           static_cast<int>(intersects[geometry_id]),
           static_cast<int>(local_idx));

    if (intersects[geometry_id]) {
      distances[geometry_id] = 0.0f;
      continue;
    }

    auto num_segment_this_multilinestring =
      multilinestrings.multilinestring_segment_count_begin()[geometry_id];
    auto multilinestring_segment_id =
      local_idx % num_segment_this_multilinestring + multilinestrings_segment_offsets[geometry_id];
    auto multipolygon_segment_id =
      local_idx / num_segment_this_multilinestring + multipolygons_segment_offsets[geometry_id];

    printf(
      "multilinestring_segment_id: %d, "
      "multipolygon_segment_id: %d\n",
      static_cast<int>(multilinestring_segment_id),
      static_cast<int>(multipolygon_segment_id));

    auto [a, b] = multilinestrings.segment_begin()[multilinestring_segment_id];
    auto [c, d] = multipolygons.segment_begin()[multipolygon_segment_id];

    auto distance = sqrt(squared_segment_distance(a, b, c, d));
    printf("ab: (%f, %f) -> (%f, %f), cd: (%f, %f) -> (%f, %f), dist: %f\n",
           static_cast<float>(a.x),
           static_cast<float>(a.y),
           static_cast<float>(b.x),
           static_cast<float>(b.y),
           static_cast<float>(c.x),
           static_cast<float>(c.y),
           static_cast<float>(d.x),
           static_cast<float>(d.y),
           static_cast<float>(distance));

    atomicMin(&distances[geometry_id], sqrt(squared_segment_distance(a, b, c, d)));
  }
}

template <typename index_t>
struct functor {
  index_t __device__ operator()(thrust::tuple<index_t, index_t> t)
  {
    return thrust::get<0>(t) * thrust::get<1>(t);
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

  // Compute the "boundary" of threads. Threads are partitioned based on the number of linestrings
  // times the number of polygons in a multipoint-multipolygon pair.
  auto segment_count_product_it = thrust::make_transform_iterator(
    thrust::make_zip_iterator(multilinestrings.multilinestring_segment_count_begin(),
                              multipolygons.multipolygon_segment_count_begin()),
    detail::functor<index_t>{});

  auto thread_bounds = rmm::device_uvector<index_t>(multilinestrings.size() + 1, stream);
  detail::zero_data_async(thread_bounds.begin(), thread_bounds.end(), stream);

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         segment_count_product_it,
                         segment_count_product_it + thread_bounds.size() - 1,
                         thrust::next(thread_bounds.begin()));

  auto multilinestring_segment_offsets =
    rmm::device_uvector<index_t>(multilinestrings.num_multilinestrings() + 1, stream);
  auto multipolygon_segment_offsets =
    rmm::device_uvector<index_t>(multipolygons.num_multipolygons() + 1, stream);

  thrust::exclusive_scan(
    rmm::exec_policy(stream),
    multilinestrings.multilinestring_segment_count_begin(),
    multilinestrings.multilinestring_segment_count_begin() + multilinestring_segment_offsets.size(),
    multilinestring_segment_offsets.begin());

  thrust::exclusive_scan(
    rmm::exec_policy(stream),
    multipolygons.multipolygon_segment_count_begin(),
    multipolygons.multipolygon_segment_count_begin() + multipolygon_segment_offsets.size(),
    multipolygon_segment_offsets.begin());

  std::cout << "multipoint intersect size: " << multipoint_intersects.size() << std::endl;
  cuspatial::test::print_device_vector(multipoint_intersects, "multipoint_intersects: ");
  cuspatial::test::print_device_vector(thread_bounds, "thread_bounds: ");
  cuspatial::test::print_device_vector(multilinestring_segment_offsets,
                                       "multilinestring_segment_offsets:");
  cuspatial::test::print_device_vector(multipolygon_segment_offsets,
                                       "multipolygon_segment_offsets: ");

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
