/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/experimental/detail/linestring_intersection_count.cuh>
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

#include <utility>

namespace cuspatial {
namespace detail {

/// Internal structure to provide convenient access to the intersection intermediate id arrays.
template <typename IntegerRange>
struct id_ranges {
  using index_t = typename IntegerRange::value_type;
  IntegerRange lhs_linestring_ids;
  IntegerRange lhs_segment_ids;
  IntegerRange rhs_linestring_ids;
  IntegerRange rhs_segment_ids;

  id_ranges(IntegerRange lhs_linestring_ids,
            IntegerRange lhs_segment_ids,
            IntegerRange rhs_linestring_ids,
            IntegerRange rhs_segment_ids)
    : lhs_linestring_ids(lhs_linestring_ids),
      lhs_segment_ids(lhs_segment_ids),
      rhs_linestring_ids(rhs_linestring_ids),
      rhs_segment_ids(rhs_segment_ids)
  {
  }

  /// Row-wise getter to the id arrays
  template <typename IndexType>
  thrust::tuple<index_t, index_t, index_t, index_t> __device__ operator[](IndexType i)
  {
    return thrust::make_tuple(
      lhs_linestring_ids[i], lhs_segment_ids[i], rhs_linestring_ids[i], rhs_segment_ids[i]);
  }

  /// Row-wise setter to the id arrays
  template <typename IndexType>
  void __device__ set(IndexType i,
                      index_t lhs_linestring_id,
                      index_t lhs_segment_id,
                      index_t rhs_linestring_id,
                      index_t rhs_segment_id)
  {
    lhs_linestring_ids[i] = lhs_linestring_id;
    lhs_segment_ids[i]    = lhs_segment_id;
    rhs_linestring_ids[i] = rhs_linestring_id;
    rhs_segment_ids[i]    = rhs_segment_id;
  }
};

/**
 * @brief Intermediate result for linestring intersection
 *
 * Analogous to arrow type: List<Geom>, where "Geom" can be point/segment.
 * Underlying id arrays have the same size as the geometry array.
 *
 * @tparam GeomType Type of geometry
 * @tparam index_t  Type of index
 */
template <typename GeomType, typename index_t>
struct linestring_intersection_intermediates {
  /// Offset array to geometries, temporary.
  std::unique_ptr<rmm::device_uvector<index_t>> offsets;
  /// Array to store the resulting geometry, non-temporary.
  std::unique_ptr<rmm::device_uvector<GeomType>> geoms;
  /// Look-back ids for the resulting geometry, temporary.
  std::unique_ptr<rmm::device_uvector<index_t>> lhs_linestring_ids;
  /// Look-back ids for the resulting geometry, temporary.
  std::unique_ptr<rmm::device_uvector<index_t>> lhs_segment_ids;
  /// Look-back ids for the resulting geometry, temporary.
  std::unique_ptr<rmm::device_uvector<index_t>> rhs_linestring_ids;
  /// Look-back ids for the resulting geometry, temporary.
  std::unique_ptr<rmm::device_uvector<index_t>> rhs_segment_ids;

  linestring_intersection_intermediates(std::size_t num_pairs,
                                        std::size_t num_geoms,
                                        rmm::device_uvector<index_t> const& num_geoms_per_pair,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
    : offsets(std::make_unique<rmm::device_uvector<index_t>>(num_pairs + 1, stream)),
      geoms(std::make_unique<rmm::device_uvector<GeomType>>(num_geoms, stream, mr)),
      lhs_linestring_ids(std::make_unique<rmm::device_uvector<index_t>>(num_geoms, stream)),
      lhs_segment_ids(std::make_unique<rmm::device_uvector<index_t>>(num_geoms, stream)),
      rhs_linestring_ids(std::make_unique<rmm::device_uvector<index_t>>(num_geoms, stream)),
      rhs_segment_ids(std::make_unique<rmm::device_uvector<index_t>>(num_geoms, stream))
  {
    // compute offsets from num_geoms_per_pair
    thrust::uninitialized_fill_n(rmm::exec_policy(stream), offsets->begin(), offsets->size(), 0);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           num_geoms_per_pair.begin(),
                           num_geoms_per_pair.end(),
                           thrust::next(offsets->begin()));
  }

  /// Return range to offset array
  auto offset_range() { return range{offsets->begin(), offsets->end()}; }

  /// Return range to geometry array
  auto geom_range() { return range{geoms->begin(), geoms->end()}; }

  /// Return id_range structure to id arrays
  auto get_id_ranges()
  {
    return id_ranges{range(lhs_linestring_ids->begin(), lhs_linestring_ids->end()),
                     range(lhs_segment_ids->begin(), lhs_segment_ids->end()),
                     range(rhs_linestring_ids->begin(), rhs_linestring_ids->end()),
                     range(rhs_segment_ids->begin(), rhs_segment_ids->end())};
  }

  /// Return the number of pairs in the intermediates
  auto size() { return offsets->size() - 1; }
};

/**
 * @brief Kernel to compute intersection and store the result and look-back indices to outputs.
 *
 * Each thread operates on one segment in `multilinestrings1`, iterates over all segments in the
 * other multilinestring.
 */
template <typename MultiLinestringRange1,
          typename MultiLinestringRange2,
          typename TempIt1,
          typename TempIt2,
          typename Offsets1,
          typename Offsets2,
          typename Offsets3,
          typename IdRanges,
          typename OutputIt1,
          typename OutputIt2>
void __global__ pairwise_linestring_intersection_simple(MultiLinestringRange1 multilinestrings1,
                                                        MultiLinestringRange2 multilinestrings2,
                                                        TempIt1 n_points_stored,
                                                        TempIt2 n_segments_stored,
                                                        Offsets1 num_points_offsets_first,
                                                        Offsets2 num_segments_offsets_first,
                                                        Offsets3 num_points_per_pair_first,
                                                        IdRanges point_ids_range,
                                                        IdRanges segment_ids_range,
                                                        OutputIt1 points_first,
                                                        OutputIt2 segments_first)
{
  using T       = typename MultiLinestringRange1::element_t;
  using types_t = uint8_t;
  using count_t = iterator_value_type<Offsets1>;
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings1.num_points();
       idx += gridDim.x * blockDim.x) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    auto const lhs_linestring_idx = multilinestrings1.intra_part_idx(part_idx);
    auto const lhs_segment_idx    = multilinestrings1.intra_point_idx(idx);
    auto [a, b]                   = multilinestrings1.segment(idx);
    auto const geometry_idx       = multilinestrings1.geometry_idx_from_part_idx(part_idx);
    auto const multilinestring2   = multilinestrings2[geometry_idx];
    auto const geometry_collection_offset =
      num_points_offsets_first[geometry_idx] + num_segments_offsets_first[geometry_idx];

    for (auto rhs_linestring_idx = 0; rhs_linestring_idx < multilinestring2.size();
         ++rhs_linestring_idx) {
      auto const linestring2 = multilinestring2[rhs_linestring_idx];
      for (auto rhs_segment_idx = 0; rhs_segment_idx < linestring2.num_segments();
           ++rhs_segment_idx) {
        auto [c, d]                   = linestring2.segment(rhs_segment_idx);
        auto [point_opt, segment_opt] = segment_intersection(segment<T>{a, b}, segment<T>{c, d});

        // Writes geometry and origin IDs to output. Note that for each pair, intersecting
        // points always precedes overlapping segments (arbitrarily).
        if (point_opt.has_value()) {
          auto r              = cuda::atomic_ref<count_t>{n_points_stored[geometry_idx]};
          auto next_point_idx = r.fetch_add(1);
          points_first[num_points_offsets_first[geometry_idx] + next_point_idx] = point_opt.value();
          auto union_column_idx = geometry_collection_offset + next_point_idx;
          auto point_column_idx = num_points_offsets_first[geometry_idx] + next_point_idx;
          point_ids_range.set(point_column_idx,
                              lhs_linestring_idx,
                              lhs_segment_idx,
                              rhs_linestring_idx,
                              rhs_segment_idx);
        } else if (segment_opt.has_value()) {
          auto r                = cuda::atomic_ref<count_t>{n_segments_stored[geometry_idx]};
          auto next_segment_idx = r.fetch_add(1);
          segments_first[num_segments_offsets_first[geometry_idx] + next_segment_idx] =
            segment_opt.value();
          auto union_column_idx =
            geometry_collection_offset + num_points_per_pair_first[geometry_idx] + next_segment_idx;
          auto segment_column_idx = num_segments_offsets_first[geometry_idx] + next_segment_idx;
          segment_ids_range.set(segment_column_idx,
                                lhs_linestring_idx,
                                lhs_segment_idx,
                                rhs_linestring_idx,
                                rhs_segment_idx);
        }
      }
    }
  }
}

/**
 * @brief Compute intersections between multilnestrings with duplicates.
 */
template <typename index_t,
          typename T,
          typename MultiLinestringRange1,
          typename MultiLinestringRange2>
std::pair<linestring_intersection_intermediates<vec_2d<T>, index_t>,
          linestring_intersection_intermediates<segment<T>, index_t>>
pairwise_linestring_intersection_with_duplicate(MultiLinestringRange1 multilinestrings1,
                                                MultiLinestringRange2 multilinestrings2,
                                                rmm::mr::device_memory_resource* mr,
                                                rmm::cuda_stream_view stream)
{
  static_assert(std::is_integral_v<index_t>, "Index type must be integral.");
  static_assert(std::is_floating_point_v<T>, "Coordinate type must be floating point.");

  auto const num_pairs = multilinestrings1.size();
  // Compute the upper bound of spaces required to store intersection results.
  rmm::device_uvector<index_t> num_points_per_pair(num_pairs, stream);
  rmm::device_uvector<index_t> num_segments_per_pair(num_pairs, stream);

  thrust::uninitialized_fill_n(rmm::exec_policy(stream), num_points_per_pair.begin(), num_pairs, 0);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_segments_per_pair.begin(), num_pairs, 0);

  detail::pairwise_linestring_intersection_upper_bound_count(multilinestrings1,
                                                             multilinestrings2,
                                                             num_points_per_pair.begin(),
                                                             num_segments_per_pair.begin(),
                                                             stream);

  // Initialize the intermediates structure
  auto num_points = thrust::reduce(
    rmm::exec_policy(stream), num_points_per_pair.begin(), num_points_per_pair.end());
  auto num_segments = thrust::reduce(
    rmm::exec_policy(stream), num_segments_per_pair.begin(), num_segments_per_pair.end());

  detail::linestring_intersection_intermediates<vec_2d<T>, index_t> points(
    num_pairs, num_points, num_points_per_pair, stream, mr);
  detail::linestring_intersection_intermediates<segment<T>, index_t> segments(
    num_pairs, num_segments, num_segments_per_pair, stream, mr);

  // Allocate a temporary vector for each thread to keep track of the number of results
  // from the current multilinestring pair has written.
  rmm::device_uvector<index_t> num_points_stored_temp(num_pairs, stream);
  rmm::device_uvector<index_t> num_segments_stored_temp(num_pairs, stream);

  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_points_stored_temp.begin(), num_pairs, 0);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_segments_stored_temp.begin(), num_pairs, 0);

  // Compute the intersections
  auto [threads_per_block, num_blocks] = grid_1d(multilinestrings1.num_points());

  detail::
    pairwise_linestring_intersection_simple<<<num_blocks, threads_per_block, 0, stream.value()>>>(
      multilinestrings1,
      multilinestrings2,
      num_points_stored_temp.begin(),
      num_segments_stored_temp.begin(),
      points.offsets->begin(),
      segments.offsets->begin(),
      num_points_per_pair.begin(),
      points.get_id_ranges(),
      segments.get_id_ranges(),
      points.geoms->begin(),
      segments.geoms->begin());

  return {std::move(points), std::move(segments)};
}

}  // namespace detail
}  // namespace cuspatial
