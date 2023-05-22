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

#include <cuspatial/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/tabulate.h>
#include <thrust/zip_function.h>

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
 * @brief Compute whether each multipoint intersects with the corresponding multipolygon.
 *
 * First, compute the point-multipolygon intersection. Then use reduce-by-key to
 * compute the multipoint-multipolygon intersection.
 * Caveat: may have load unbalanced kernel if input is skewed with non-uniform distribution with the
 * number of polygons in multipolygon.
 *
 * @tparam MultiPointRange An instantiation of multipoint_range
 * @tparam MultiPolygonRange An instantiation of multipolygon_range
 * @param multipoints The range to the multipoints to compute
 * @param multipolygons The range to the multipolygons to test
 * @param stream The CUDA stream on which to perform computations
 * @return A uint8_t array, `1` if the multipoint intersects with the multipolygon, `0` otherwise.
 */
template <typename MultiPointRange, typename MultiPolygonRange>
rmm::device_uvector<uint8_t> point_polygon_intersects(MultiPointRange multipoints,
                                                      MultiPolygonRange multipolygons,
                                                      rmm::cuda_stream_view stream)
{
  using index_t = iterator_value_type<typename MultiPointRange::geometry_it_t>;
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
                        thrust::equal_to<index_t>(),
                        thrust::logical_or<uint8_t>());

  return multipoint_intersects;
}

/**
 * @brief Compute the thread bound between two ranges of partitions
 *
 * @tparam CountIterator1
 * @tparam CountIterator2
 * @param lhs
 * @param rhs
 * @param stream
 * @return rmm::device_uvector<IndexType>
 */
template <typename CountIterator1,
          typename CountIterator2,
          typename index_t = iterator_value_type<CountIterator1>>
rmm::device_uvector<index_t> compute_segment_thread_bounds(CountIterator1 lhs_begin,
                                                           CountIterator1 lhs_end,
                                                           CountIterator2 rhs_begin,
                                                           rmm::cuda_stream_view stream)
{
  auto size = thrust::distance(lhs_begin, lhs_end) + 1;

  // Compute the "boundary" of threads. Threads are partitioned based on the number of linestrings
  // times the number of polygons in a multilinestring-multipolygon pair.
  auto segment_count_product_it =
    thrust::make_transform_iterator(thrust::make_zip_iterator(lhs_begin, rhs_begin),
                                    thrust::make_zip_function(thrust::multiplies<index_t>{}));

  // Computes the "thread boundary" of each pair. This array partitions the thread range by
  // geometries. E.g. threadIdx within [thread_bounds[i], thread_bounds[i+1]) computes distances of
  // the ith pair.
  auto thread_bounds = rmm::device_uvector<index_t>(size, stream);
  detail::zero_data_async(thread_bounds.begin(), thread_bounds.end(), stream);

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         segment_count_product_it,
                         segment_count_product_it + thread_bounds.size() - 1,
                         thrust::next(thread_bounds.begin()));

  return thread_bounds;
}

}  // namespace detail
}  // namespace cuspatial
