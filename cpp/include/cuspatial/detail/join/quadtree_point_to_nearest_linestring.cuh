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

#include <cuspatial/detail/algorithm/point_linestring_distance.cuh>
#include <cuspatial/detail/join/get_quad_and_local_point_indices.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/point_quadtree.cuh>
#include <cuspatial/range/multilinestring_range.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <cstdint>

namespace cuspatial {
namespace detail {

template <typename QuadOffsetsIter>
inline __device__ std::pair<uint32_t, uint32_t> get_local_linestring_index_and_count(
  uint32_t const linestring_index, QuadOffsetsIter quad_offsets, QuadOffsetsIter quad_offsets_end)
{
  auto const lhs_end     = quad_offsets;
  auto const rhs_end     = quad_offsets_end;
  auto const quad_offset = quad_offsets[linestring_index];
  auto const lhs =
    thrust::lower_bound(thrust::seq, lhs_end, quad_offsets + linestring_index, quad_offset);
  auto const rhs =
    thrust::upper_bound(thrust::seq, quad_offsets + linestring_index, rhs_end, quad_offset);

  return std::make_pair(
    // local_linestring_index
    static_cast<uint32_t>(thrust::distance(lhs, quad_offsets + linestring_index)),
    // num_linestrings_in_quad
    static_cast<uint32_t>(thrust::distance(lhs, rhs)));
}

template <typename QuadOffsetsIter, typename QuadLengthsIter>
inline __device__ std::pair<uint32_t, uint32_t> get_transposed_point_and_pair_index(
  uint32_t const global_index,
  uint32_t const* point_offsets,
  uint32_t const* point_offsets_end,
  QuadOffsetsIter quad_offsets,
  QuadOffsetsIter quad_offsets_end,
  QuadLengthsIter quad_lengths)
{
  auto const [quad_linestring_index, local_point_index] =
    get_quad_and_local_point_indices(global_index, point_offsets, point_offsets_end);

  auto const [local_linestring_index, num_linestrings_in_quad] =
    get_local_linestring_index_and_count(quad_linestring_index, quad_offsets, quad_offsets_end);

  auto const quad_point_offset           = quad_offsets[quad_linestring_index];
  auto const num_points_in_quad          = quad_lengths[quad_linestring_index];
  auto const quad_linestring_offset      = quad_linestring_index - local_linestring_index;
  auto const quad_linestring_point_start = local_linestring_index * num_points_in_quad;
  auto const transposed_point_start      = quad_linestring_point_start + local_point_index;

  return std::make_pair(
    // transposed point index
    (transposed_point_start / num_linestrings_in_quad) + quad_point_offset,
    // transposed linestring index
    (transposed_point_start % num_linestrings_in_quad) + quad_linestring_offset);
}

template <typename PointIter,
          typename PointOffsetsIter,
          typename QuadOffsetsIter,
          typename QuadLengthsIter,
          typename LinestringIndexIterator,
          class MultiLinestringRange,
          typename T = cuspatial::iterator_vec_base_type<PointIter>>
struct compute_point_linestring_indices_and_distances {
  PointIter points;
  PointOffsetsIter point_offsets;
  PointOffsetsIter point_offsets_end;
  QuadOffsetsIter quad_offsets;
  QuadOffsetsIter quad_offsets_end;
  QuadLengthsIter quad_lengths;
  LinestringIndexIterator linestring_indices;
  MultiLinestringRange linestrings;

  compute_point_linestring_indices_and_distances(PointIter points,
                                                 PointOffsetsIter point_offsets,
                                                 PointOffsetsIter point_offsets_end,
                                                 QuadOffsetsIter quad_offsets,
                                                 QuadOffsetsIter quad_offsets_end,
                                                 QuadLengthsIter quad_lengths,
                                                 LinestringIndexIterator linestring_indices,
                                                 MultiLinestringRange linestrings)
    : points(points),
      point_offsets(point_offsets),
      point_offsets_end(point_offsets_end),
      quad_offsets(quad_offsets),
      quad_offsets_end(quad_offsets_end),
      quad_lengths(quad_lengths),
      linestring_indices(linestring_indices),
      linestrings(linestrings)
  {
  }

  inline __device__ thrust::tuple<uint32_t, uint32_t, T> operator()(uint32_t const global_index)
  {
    auto const [point_id, linestring_id] = get_transposed_point_and_pair_index(
      global_index, point_offsets, point_offsets_end, quad_offsets, quad_offsets_end, quad_lengths);

    // We currently support only single-linestring multilinestrings, so use the zero index
    auto linestring = linestrings[linestring_indices[linestring_id]][0];
    auto const distance =
      point_linestring_distance(thrust::raw_reference_cast(points[point_id]), linestring);

    return thrust::make_tuple(point_id, linestring_indices[linestring_id], distance);
  }
};

}  // namespace detail

template <class LinestringIndexIterator,
          class QuadIndexIterator,
          class PointIndexIterator,
          class PointIterator,
          class MultiLinestringRange,
          typename IndexType,
          typename T>
std::tuple<rmm::device_uvector<IndexType>, rmm::device_uvector<IndexType>, rmm::device_uvector<T>>
quadtree_point_to_nearest_linestring(LinestringIndexIterator linestring_indices_first,
                                     LinestringIndexIterator linestring_indices_last,
                                     QuadIndexIterator quad_indices_first,
                                     point_quadtree_ref quadtree,
                                     PointIndexIterator point_indices_first,
                                     PointIndexIterator point_indices_last,
                                     PointIterator points_first,
                                     MultiLinestringRange linestrings,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUSPATIAL_EXPECTS(linestrings.num_multilinestrings() == linestrings.num_linestrings(),
                    "Only one linestring per multilinestring currently supported.");

  auto num_linestring_quad_pairs = std::distance(linestring_indices_first, linestring_indices_last);

  auto quad_lengths_iter =
    thrust::make_permutation_iterator(quadtree.length_begin(), quad_indices_first);

  auto quad_offsets_iter =
    thrust::make_permutation_iterator(quadtree.offset_begin(), quad_indices_first);

  // Compute a "local" set of zero-based point offsets from number of points in each quadrant
  // Use `num_poly_quad_pairs + 1` as the length so that the last element produced by
  // `inclusive_scan` is the total number of points to be tested against any polygon.
  rmm::device_uvector<IndexType> local_point_offsets(num_linestring_quad_pairs + 1, stream);

  // inclusive scan of quad_lengths_iter
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         quad_lengths_iter,
                         quad_lengths_iter + num_linestring_quad_pairs,
                         local_point_offsets.begin() + 1);

  // Ensure local point offsets starts at 0
  IndexType init{0};
  local_point_offsets.set_element_async(0, init, stream);

  // The last element is the total number of points to test against any polygon.
  auto num_point_linestring_pairs = local_point_offsets.back_element(stream);

  // Enumerate the point X/Ys using the sorted `point_indices` (from quadtree construction)
  auto point_xys_iter = thrust::make_permutation_iterator(points_first, point_indices_first);

  //
  // Compute the combination of point and linestring index pairs. For each linestring / quadrant
  // pair, enumerate pairs of (point_index, linestring_index) for each point in each quadrant,
  // and calculate the minimum distance between each point / linestring pair.
  //
  // In Python pseudocode:
  // ```
  // pl_pairs_and_dist = []
  // for linestring, quadrant in lq_pairs:
  //   for point in quadrant:
  //     pl_pairs_and_dist.append((point, linestring, min_distance(point, linestring)))
  // ```
  //
  // However, the above pseudocode produces values in an order such that the distance
  // from a point to each linestring cannot be reduced with `thrust::reduce_by_key`:
  // ```
  //   point | linestring | distance
  //       0 |          0 |     10.0
  //       1 |          0 |     30.0
  //       2 |          0 |     20.0
  //       0 |          1 |     30.0
  //       1 |          1 |     20.0
  //       2 |          1 |     10.0
  // ```
  //
  // In order to use `thrust::reduce_by_key` to compute the minimum distance from a point to
  // the linestrings in its quadrant, the above table needs to be sorted by `point` instead of
  // `linestring`:
  // ```
  //   point | linestring | distance
  //       0 |          0 |     10.0
  //       0 |          1 |     30.0
  //       1 |          0 |     30.0
  //       1 |          1 |     20.0
  //       2 |          0 |     20.0
  //       2 |          1 |     10.0
  // ```
  //
  // A naive approach would be to allocate memory for the above three columns, sort the
  // columns by `point`, then use `thrust::reduce_by_key` to compute the min distances.
  //
  // The sizes of the intermediate buffers required can easily grow beyond available
  // device memory, so a better approach is to use a Thrust iterator to yield values
  // in the sorted order as we do here.
  //
  auto all_point_linestring_indices_and_distances = detail::make_counting_transform_iterator(
    0u,
    compute_point_linestring_indices_and_distances{point_xys_iter,
                                                   local_point_offsets.begin(),
                                                   local_point_offsets.end(),
                                                   quad_offsets_iter,
                                                   quad_offsets_iter + num_linestring_quad_pairs,
                                                   quad_lengths_iter,
                                                   linestring_indices_first,
                                                   linestrings});

  auto all_point_indices =
    thrust::make_transform_iterator(all_point_linestring_indices_and_distances,
                                    cuda::proclaim_return_type<uint32_t>(
                                      [] __device__(auto const& x) { return thrust::get<0>(x); }));

  // Allocate vectors for the distances min reduction
  auto num_points = std::distance(point_indices_first, point_indices_last);
  rmm::device_uvector<uint32_t> point_idxs(num_points, stream);  // temporary, used to scatter

  rmm::device_uvector<uint32_t> output_linestring_idxs(num_points, stream, mr);
  rmm::device_uvector<T> output_distances(num_points, stream, mr);
  rmm::device_uvector<uint32_t> output_point_idxs(num_points, stream, mr);

  // Fill distances with 0
  zero_data_async(output_distances.begin(), output_distances.end(), stream);
  // Reduce the intermediate point/linestring indices to lists of point/linestring index pairs
  // and distances, selecting the linestring index closest to each point.
  auto point_idxs_end = thrust::reduce_by_key(
    rmm::exec_policy(stream),
    all_point_indices,  // point indices in
    all_point_indices + num_point_linestring_pairs,
    all_point_linestring_indices_and_distances,
    point_idxs.begin(),  // point indices out
    // point/linestring indices and distances out
    thrust::make_zip_iterator(
      thrust::make_discard_iterator(), output_linestring_idxs.begin(), output_distances.begin()),
    thrust::equal_to<uint32_t>(),  // comparator
    // binop to select the point/linestring pair with the smallest distance
    cuda::proclaim_return_type<thrust::tuple<uint32_t, uint32_t, T>>(
      [] __device__(auto const& lhs, auto const& rhs) {
        T const& d_lhs = thrust::get<2>(lhs);
        T const& d_rhs = thrust::get<2>(rhs);
        // If lhs distance is 0, choose rhs
        if (d_lhs == T{0}) { return rhs; }
        // if rhs distance is 0, choose lhs
        if (d_rhs == T{0}) { return lhs; }
        // If distances to lhs/rhs are the same, choose linestring with smallest id
        if (d_lhs == d_rhs) {
          auto const& i_lhs = thrust::get<1>(lhs);
          auto const& i_rhs = thrust::get<1>(rhs);
          return i_lhs < i_rhs ? lhs : rhs;
        }
        // Otherwise choose linestring with smallest distance
        return d_lhs < d_rhs ? lhs : rhs;
      }));

  auto const num_distances = thrust::distance(point_idxs.begin(), point_idxs_end.first);

  auto point_linestring_idxs_and_distances = thrust::make_zip_iterator(
    point_idxs.begin(), output_linestring_idxs.begin(), output_distances.begin());

  // scatter the values from their positions after reduction into their output positions
  thrust::scatter(
    rmm::exec_policy(stream),
    point_linestring_idxs_and_distances,
    point_linestring_idxs_and_distances + num_distances,
    point_idxs.begin(),
    thrust::make_zip_iterator(
      output_point_idxs.begin(), output_linestring_idxs.begin(), output_distances.begin()));

  return std::tuple{
    std::move(output_point_idxs), std::move(output_linestring_idxs), std::move(output_distances)};
}

}  // namespace cuspatial
