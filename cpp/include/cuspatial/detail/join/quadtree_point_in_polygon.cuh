/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuspatial/detail/join/get_quad_and_local_point_indices.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/point_quadtree.cuh>
#include <cuspatial/range/multipolygon_range.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <limits>

namespace cuspatial {
namespace detail {

template <class QuadOffsetsIterator, class PointOffsetsIterator, class PolyIndexIterator>
struct compute_poly_and_point_indices {
  QuadOffsetsIterator quad_point_offsets_begin;
  PointOffsetsIterator point_offsets_begin;
  PointOffsetsIterator point_offsets_end;
  PolyIndexIterator poly_indices_begin;

  compute_poly_and_point_indices(QuadOffsetsIterator quad_point_offsets_begin,
                                 PointOffsetsIterator point_offsets_begin,
                                 PointOffsetsIterator point_offsets_end,
                                 PolyIndexIterator poly_indices_begin)
    : quad_point_offsets_begin(quad_point_offsets_begin),
      point_offsets_begin(point_offsets_begin),
      point_offsets_end(point_offsets_end),
      poly_indices_begin(poly_indices_begin)
  {
  }

  using IndexType = iterator_value_type<QuadOffsetsIterator>;

  inline thrust::tuple<IndexType, IndexType> __device__
  operator()(std::uint64_t const global_index) const
  {
    auto const [quad_poly_index, local_point_index] =
      get_quad_and_local_point_indices(global_index, point_offsets_begin, point_offsets_end);
    auto const point_idx = quad_point_offsets_begin[quad_poly_index] + local_point_index;
    auto const poly_idx  = poly_indices_begin[quad_poly_index];
    return thrust::make_tuple(poly_idx, point_idx);
  }
};

template <class PointIterator, class MultiPolygonRange>
struct test_poly_point_intersection {
  using T         = iterator_vec_base_type<PointIterator>;
  using IndexType = iterator_value_type<typename MultiPolygonRange::part_it_t>;

  test_poly_point_intersection(PointIterator points_first, MultiPolygonRange polygons)
    : points_first(points_first), polygons(polygons)
  {
  }

  PointIterator points_first;
  MultiPolygonRange polygons;

  inline bool __device__ operator()(thrust::tuple<IndexType, IndexType> const& poly_point_idxs)
  {
    auto const poly_idx  = thrust::get<0>(poly_point_idxs);
    auto const point_idx = thrust::get<1>(poly_point_idxs);

    vec_2d<T> const& point = points_first[point_idx];

    return is_point_in_polygon(point, polygons[poly_idx][0]);
  }
};

}  // namespace detail

template <class PolyIndexIterator,
          class QuadIndexIterator,
          class PointIndexIterator,
          class PointIterator,
          class MultiPolygonRange,
          class IndexType>
std::pair<rmm::device_uvector<IndexType>, rmm::device_uvector<IndexType>> quadtree_point_in_polygon(
  PolyIndexIterator poly_indices_first,
  PolyIndexIterator poly_indices_last,
  QuadIndexIterator quad_indices_first,
  point_quadtree_ref quadtree,
  PointIndexIterator point_indices_first,
  PointIndexIterator point_indices_last,
  PointIterator points_first,
  MultiPolygonRange polygons,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  using T = iterator_vec_base_type<PointIterator>;

  CUSPATIAL_EXPECTS(polygons.num_multipolygons() == polygons.num_polygons(),
                    "Only one polygon per multipolygon currently supported.");

  auto num_poly_quad_pairs = std::distance(poly_indices_first, poly_indices_last);

  // The quadtree length is an iterator of uint32_t, but we have to transform into uint64_t values
  // so the thrust::inclusive_scan accumulates into uint64_t outputs. Changing the output iterator
  // to uint64_t isn't sufficient to achieve this behavior.
  auto quad_lengths_iter = thrust::make_transform_iterator(
    thrust::make_permutation_iterator(quadtree.length_begin(), quad_indices_first),
    cuda::proclaim_return_type<std::uint64_t>([] __device__(IndexType const& i) -> std::uint64_t {
      return static_cast<std::uint64_t>(i);
    }));

  auto quad_offsets_iter =
    thrust::make_permutation_iterator(quadtree.offset_begin(), quad_indices_first);

  // Compute a "local" set of zero-based point offsets from the number of points in each quadrant.
  //
  // Use `num_poly_quad_pairs + 1` as the length so that the last element produced by
  // `inclusive_scan` is the total number of points to be tested against any polygon.
  //
  // Accumulate into uint64_t, because the prefix sums can overflow the size of uint32_t
  // when testing a large number of polygons against a large quadtree.
  rmm::device_uvector<std::uint64_t> local_point_offsets(num_poly_quad_pairs + 1, stream);

  // inclusive scan of quad_lengths_iter
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         quad_lengths_iter,
                         quad_lengths_iter + num_poly_quad_pairs,
                         local_point_offsets.begin() + 1);

  // Ensure local point offsets starts at 0
  std::uint64_t init{0};
  local_point_offsets.set_element_async(0, init, stream);

  // The last element is the total number of points to test against any polygon.
  auto num_total_points = local_point_offsets.back_element(stream);

  // The largest supported input size for thrust::count_if/copy_if is INT32_MAX.
  // This functor iterates over the input space and processes up to INT32_MAX elements at a time.
  std::uint64_t max_points_to_test = std::numeric_limits<std::int32_t>::max();
  auto count_in_chunks             = [&](auto const& func) {
    std::uint64_t memo{};
    for (std::uint64_t offset{0}; offset < num_total_points; offset += max_points_to_test) {
      memo += func(memo, offset, std::min(max_points_to_test, num_total_points - offset));
    }
    return memo;
  };

  detail::test_poly_point_intersection test_poly_point_pair{
    // Enumerate the point X/Ys using the sorted `point_indices` (from quadtree construction)
    thrust::make_permutation_iterator(points_first, point_indices_first),
    polygons};

  // Compute the combination of polygon and point index pairs. For each polygon/quadrant pair,
  // enumerate pairs of (poly_index, point_index) for each point in each quadrant.
  //
  // In Python pseudocode:
  // ```
  // pp_pairs = []
  // for polygon, quadrant in pq_pairs:
  //   for point in quadrant:
  //     pp_pairs.append((polygon, point))
  // ```
  //
  auto global_to_poly_and_point_indices = [&](auto offset = 0) {
    return detail::make_counting_transform_iterator(
      offset,
      detail::compute_poly_and_point_indices{quad_offsets_iter,
                                             local_point_offsets.begin(),
                                             local_point_offsets.end(),
                                             poly_indices_first});
  };

  auto run_quadtree_point_in_polygon = [&](auto output_size) {
    // Allocate the output polygon and point index pair vectors
    rmm::device_uvector<IndexType> poly_indices(output_size, stream);
    rmm::device_uvector<IndexType> point_indices(output_size, stream);

    auto num_intersections = count_in_chunks([&](auto memo, auto offset, auto size) {
      auto poly_and_point_indices =
        thrust::make_zip_iterator(poly_indices.begin(), point_indices.begin()) + memo;
      // Remove (poly, point) pairs that don't intersect
      return thrust::distance(poly_and_point_indices,
                              thrust::copy_if(rmm::exec_policy(stream),
                                              global_to_poly_and_point_indices(offset),
                                              global_to_poly_and_point_indices(offset) + size,
                                              poly_and_point_indices,
                                              test_poly_point_pair));
    });

    if (num_intersections < output_size) {
      poly_indices.resize(num_intersections, stream);
      point_indices.resize(num_intersections, stream);
      poly_indices.shrink_to_fit(stream);
      point_indices.shrink_to_fit(stream);
    }

    return std::pair{std::move(poly_indices), std::move(point_indices)};
  };

  try {
    // First attempt to run the hit test assuming allocating space for all possible intersections
    // fits into the available memory.
    return run_quadtree_point_in_polygon(num_total_points);
  } catch (rmm::out_of_memory const&) {
    // If we OOM the first time, pre-compute the number of hits and allocate only that amount of
    // space for the output buffers. This halves performance, but it should at least return valid
    // results.
    return run_quadtree_point_in_polygon(count_in_chunks([&](auto memo, auto offset, auto size) {
      return thrust::count_if(rmm::exec_policy(stream),
                              global_to_poly_and_point_indices(offset),
                              global_to_poly_and_point_indices(offset) + size,
                              test_poly_point_pair);
    }));
  }
}

}  // namespace cuspatial
