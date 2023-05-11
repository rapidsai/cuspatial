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

}  // namespace detail
}  // namespace cuspatial
