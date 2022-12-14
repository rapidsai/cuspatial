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
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/experimental/detail/linestring_intersection_count.cuh>
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

#include <type_traits>
#include <utility>

namespace cuspatial {
namespace detail {

namespace intersection_functors {

template <typename Keys, typename Values>
struct offsets_update_functor {
  Keys reduced_keys_begin;
  Keys reduced_keys_end;
  Values reduced_values_begin;
  Values reduced_values_end;

  offsets_update_functor(Keys kb, Keys ke, Values vb, Values ve)
    : reduced_keys_begin(kb), reduced_keys_end(ke), reduced_values_begin(vb), reduced_values_end(ve)
  {
  }

  int __device__ operator()(int offset, int i)
  {
    if (i == 0) return 0;
    auto j = thrust::distance(
      reduced_keys_begin,
      thrust::upper_bound(thrust::seq, reduced_keys_begin, reduced_keys_end, i - 1));
    return offset - reduced_values_begin[j - 1];
  }
};

template <typename Iterator>
struct offsets_to_keys_functor {
  Iterator _offsets_begin;
  Iterator _offsets_end;

  offsets_to_keys_functor(Iterator offset_begin, Iterator offset_end)
    : _offsets_begin(offset_begin), _offsets_end(offset_end)
  {
  }

  template <typename IndexType>
  IndexType __device__ operator()(IndexType i)
  {
    return thrust::distance(
      _offsets_begin,
      thrust::prev(thrust::upper_bound(thrust::seq, _offsets_begin, _offsets_end, i)));
  }
};

}  // namespace intersection_functors

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
template <typename GeomType, typename IndexType>
struct linestring_intersection_intermediates {
  using geometry_t = GeomType;
  using index_t    = IndexType;

  /// Offset array to geometries, temporary.
  std::unique_ptr<rmm::device_uvector<IndexType>> offsets;
  /// Array to store the resulting geometry, non-temporary.
  std::unique_ptr<rmm::device_uvector<GeomType>> geoms;
  /// Look-back ids for the resulting geometry, temporary.
  std::unique_ptr<rmm::device_uvector<IndexType>> lhs_linestring_ids;
  /// Look-back ids for the resulting geometry, temporary.
  std::unique_ptr<rmm::device_uvector<IndexType>> lhs_segment_ids;
  /// Look-back ids for the resulting geometry, temporary.
  std::unique_ptr<rmm::device_uvector<IndexType>> rhs_linestring_ids;
  /// Look-back ids for the resulting geometry, temporary.
  std::unique_ptr<rmm::device_uvector<IndexType>> rhs_segment_ids;

  linestring_intersection_intermediates(
    std::unique_ptr<rmm::device_uvector<index_t>> offsets,
    std::unique_ptr<rmm::device_uvector<GeomType>> geoms,
    std::unique_ptr<rmm::device_uvector<index_t>> lhs_linestring_ids,
    std::unique_ptr<rmm::device_uvector<index_t>> lhs_segment_ids,
    std::unique_ptr<rmm::device_uvector<index_t>> rhs_linestring_ids,
    std::unique_ptr<rmm::device_uvector<index_t>> rhs_segment_ids)
    : offsets(std::move(offsets)),
      geoms(std::move(geoms)),
      lhs_linestring_ids(std::move(lhs_linestring_ids)),
      lhs_segment_ids(std::move(lhs_segment_ids)),
      rhs_linestring_ids(std::move(rhs_linestring_ids)),
      rhs_segment_ids(std::move(rhs_segment_ids))
  {
  }

  linestring_intersection_intermediates(std::size_t num_pairs,
                                        std::size_t num_geoms,
                                        rmm::device_uvector<IndexType> const& num_geoms_per_pair,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
    : offsets(std::make_unique<rmm::device_uvector<IndexType>>(num_pairs + 1, stream)),
      geoms(std::make_unique<rmm::device_uvector<GeomType>>(num_geoms, stream, mr)),
      lhs_linestring_ids(std::make_unique<rmm::device_uvector<IndexType>>(num_geoms, stream)),
      lhs_segment_ids(std::make_unique<rmm::device_uvector<IndexType>>(num_geoms, stream)),
      rhs_linestring_ids(std::make_unique<rmm::device_uvector<IndexType>>(num_geoms, stream)),
      rhs_segment_ids(std::make_unique<rmm::device_uvector<IndexType>>(num_geoms, stream))
  {
    // compute offsets from num_geoms_per_pair

    zero_data_async(offsets->begin(), offsets->end(), stream);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           num_geoms_per_pair.begin(),
                           num_geoms_per_pair.end(),
                           thrust::next(offsets->begin()));
  }

  /// Given a flag array, remove the ith geometry if `flags[i] == 1`.
  /// @pre flag array must have the same size as the geometry array.
  template <typename FlagRange>
  void remove_if(FlagRange flags, rmm::cuda_stream_view stream)
  {
    // The offsets for the geometry array marks the start index for each list.
    // When geometries are removed, the offset for the next list must
    // be subtracted by the number of removed geometries in *all* previous lists.

    // Use `reduce_by_key` to compute the number of removed geometry per list.
    rmm::device_uvector<index_t> reduced_keys(num_pairs(), stream);
    rmm::device_uvector<index_t> reduced_flags(num_pairs(), stream);
    auto keys_begin = make_counting_transform_iterator(
      0, intersection_functors::offsets_to_keys_functor{offsets->begin(), offsets->end()});

    auto [keys_end, flags_end] =
      thrust::reduce_by_key(rmm::exec_policy(stream),
                            keys_begin,
                            keys_begin + flags.size(),
                            flags.begin(),
                            reduced_keys.begin(),
                            reduced_flags.begin(),
                            thrust::equal_to<index_t>(),
                            thrust::plus<index_t>());  // explicitly cast flags to index_t type
                                                       // before adding to avoid overflow.

    reduced_keys.resize(thrust::distance(reduced_keys.begin(), keys_end), stream);
    reduced_flags.resize(thrust::distance(reduced_flags.begin(), flags_end), stream);

    // Use `inclusive_scan` to compute the number of removed geometries in *all* previous lists.
    thrust::inclusive_scan(
      rmm::exec_policy(stream), reduced_flags.begin(), reduced_flags.end(), reduced_flags.begin());

    // Update the offsets
    thrust::transform(
      rmm::exec_policy(stream),
      offsets->begin(),
      offsets->end(),
      thrust::make_counting_iterator(0),
      offsets->begin(),
      intersection_functors::offsets_update_functor{
        reduced_keys.begin(), reduced_keys.end(), reduced_flags.begin(), reduced_flags.end()});

    // Remove the geometries and the corresponding ids per flag.
    auto geom_id_it  = thrust::make_zip_iterator(geoms->begin(),
                                                lhs_linestring_ids->begin(),
                                                lhs_segment_ids->begin(),
                                                rhs_linestring_ids->begin(),
                                                rhs_segment_ids->begin());
    auto geom_id_end = thrust::remove_if(rmm::exec_policy(stream),
                                         geom_id_it,
                                         geom_id_it + geoms->size(),
                                         flags.begin(),
                                         [] __device__(uint8_t flag) { return flag == 1; });

    auto new_geom_size = thrust::distance(geom_id_it, geom_id_end);
    geoms->resize(new_geom_size, stream);
    lhs_linestring_ids->resize(new_geom_size, stream);
    lhs_segment_ids->resize(new_geom_size, stream);
    rhs_linestring_ids->resize(new_geom_size, stream);
    rhs_segment_ids->resize(new_geom_size, stream);
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

  /// Return list-id corresponding to the geometry
  auto keys_begin()
  {
    return make_counting_transform_iterator(
      0, intersection_functors::offsets_to_keys_functor{offsets->begin(), offsets->end()});
  }

  /// Return the number of pairs in the intermediates
  auto num_pairs() { return offsets->size() - 1; }

  /// Return the number of geometries in the intermediates
  auto num_geoms() { return geoms->size(); }
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
    if (auto const part_idx_opt = multilinestrings1.part_idx_from_segment_idx(idx);
        part_idx_opt.has_value()) {
      auto const part_idx           = part_idx_opt.value();
      auto const lhs_linestring_idx = multilinestrings1.intra_part_idx(part_idx);
      auto const lhs_segment_idx    = multilinestrings1.intra_point_idx(idx);
      auto [a, b]                   = multilinestrings1.segment(idx);
      auto const geometry_idx       = multilinestrings1.geometry_idx_from_part_idx(part_idx);
      auto const multilinestring2   = multilinestrings2[geometry_idx];
      auto const geometry_collection_offset =
        num_points_offsets_first[geometry_idx] + num_segments_offsets_first[geometry_idx];

      for (auto [rhs_linestring_idx, linestring2] : multilinestring2.enumerate()) {
        for (auto [rhs_segment_idx, segment2] : linestring2.enumerate()) {
          auto [c, d]                   = segment2;
          auto [point_opt, segment_opt] = segment_intersection(segment<T>{a, b}, segment<T>{c, d});

          // Writes geometry and origin IDs to output. Note that for each pair, intersecting
          // points always precedes overlapping segments (arbitrarily).
          if (point_opt.has_value()) {
            auto next_point_idx =
              cuda::atomic_ref<count_t>{n_points_stored[geometry_idx]}.fetch_add(1);
            points_first[num_points_offsets_first[geometry_idx] + next_point_idx] =
              point_opt.value();
            auto union_column_idx = geometry_collection_offset + next_point_idx;
            auto point_column_idx = num_points_offsets_first[geometry_idx] + next_point_idx;
            point_ids_range.set(point_column_idx,
                                lhs_linestring_idx,
                                lhs_segment_idx,
                                rhs_linestring_idx,
                                rhs_segment_idx);
          } else if (segment_opt.has_value()) {
            auto next_segment_idx =
              cuda::atomic_ref<count_t>{n_segments_stored[geometry_idx]}.fetch_add(1);
            segments_first[num_segments_offsets_first[geometry_idx] + next_segment_idx] =
              segment_opt.value();
            auto union_column_idx = geometry_collection_offset +
                                    num_points_per_pair_first[geometry_idx] + next_segment_idx;
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
}

/**
 * @brief Compute intersection results between pairs of multilinestrings. The result may contain
 * duplicate points, mergeable segments and mergeable point on segments.
 */
template <typename index_t,
          typename T,
          typename MultiLinestringRange1,
          typename MultiLinestringRange2>
std::pair<linestring_intersection_intermediates<vec_2d<T>, index_t>,
          linestring_intersection_intermediates<segment<T>, index_t>>
pairwise_linestring_intersection_with_duplicates(MultiLinestringRange1 multilinestrings1,
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

  detail::zero_data_async(num_points_per_pair.begin(), num_points_per_pair.end(), stream);
  detail::zero_data_async(num_segments_per_pair.begin(), num_segments_per_pair.end(), stream);

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

  detail::zero_data_async(num_points_stored_temp.begin(), num_points_stored_temp.end(), stream);
  detail::zero_data_async(num_segments_stored_temp.begin(), num_segments_stored_temp.end(), stream);

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
