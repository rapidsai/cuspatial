/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/iterator_factory.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>
#include <thrust/uninitialized_fill.h>

#include <ranger/ranger.hpp>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @pre All segments in range @p segments , it is presorted using `segment_comparator`.
 *
 * @brief Kernel to merge segments. Each thread works on one segment space, within each space,
 * `segment_comparator` guarantees that segments with same slope are grouped together. We call
 * each of such group is a "mergeable group". Within each mergeable group, the first segment is
 * the "leading segment". The algorithm behave as follows:
 *
 * 1. For each mergeable group, loop over the rest of the segments in the group and see if it is
 * mergeable with the leading segment. If it is, overwrite the leading segment with the merged
 * result. Then mark the segment as merged by setting the flag to 1. This makes sure the inner loop
 * for each merged segment is not run again.
 * 2. Repeat 1 until all mergeable group is processed.
 */
template <typename OffsetRange, typename SegmentRange, typename OutputIt>
CUSPATIAL_KERNEL void simple_find_and_combine_segments_kernel(OffsetRange offsets,
                                                              SegmentRange segments,
                                                              OutputIt merged_flag)
{
  for (auto pair_idx : ranger::grid_stride_range(offsets.size() - 1)) {
    // Zero-initialize flags for all segments in current space.
    for (auto i = offsets[pair_idx]; i < offsets[pair_idx + 1]; i++) {
      merged_flag[i] = 0;
    }

    // For each of the segment, loop over the rest of the segment in the space and see
    // if it is mergeable with the current segment.
    // Note that if the current segment is already merged. Skip checking.
    for (auto i = offsets[pair_idx]; i < offsets[pair_idx + 1] && merged_flag[i] != 1; i++) {
      for (auto j = i + 1; j < offsets[pair_idx + 1]; j++) {
        auto res = maybe_merge_segments(segments[i], segments[j]);
        if (res.has_value()) {
          // segments[i] can be merged from segments[j]
          segments[i]    = res.value();
          merged_flag[j] = 1;
        }
      }
    }
  }
}

/**
 * @brief Comparator for sorting the segment range.
 *
 * This comparator makes sure that the segment range is sorted such that:
 * 1. Segments with the same space id are grouped together.
 * 2. Segments within the same space are grouped by their slope.
 * 3. Within each slope group, segments are sorted by their lower left point.
 */
template <typename index_t, typename T>
struct segment_comparator {
  bool __device__ operator()(thrust::tuple<index_t, segment<T>> const& lhs,
                             thrust::tuple<index_t, segment<T>> const& rhs) const
  {
    auto lhs_index   = thrust::get<0>(lhs);
    auto rhs_index   = thrust::get<0>(rhs);
    auto lhs_segment = thrust::get<1>(lhs);
    auto rhs_segment = thrust::get<1>(rhs);

    // Compare space id
    if (lhs_index == rhs_index) {
      // Compare slope
      if (lhs_segment.collinear(rhs_segment)) {
        // Sort by the lower left point of the segment.
        return lhs_segment.lower_left() < rhs_segment.lower_left();
      }
      return lhs_segment.slope() < rhs_segment.slope();
    }
    return lhs_index < rhs_index;
  }
};

/**
 * @internal
 * @brief For each pair of mergeable segment, overwrites the first segment with merged result,
 * sets the flag for the second segment as 1.
 *
 * @note This function will alter the input segment range by rearranging the order of the segments
 * within each space so that merging kernel can take place.
 */
template <typename OffsetRange, typename SegmentRange, typename OutputIt>
void find_and_combine_segment(OffsetRange offsets,
                              SegmentRange segments,
                              OutputIt merged_flag,
                              rmm::cuda_stream_view stream)
{
  using index_t   = typename OffsetRange::value_type;
  using T         = typename SegmentRange::value_type::value_type;
  auto num_spaces = offsets.size() - 1;
  if (num_spaces == 0) return;

  // Construct a key iterator using the offsets of the segment and the segment itself.
  auto space_id_iter         = make_geometry_id_iterator<index_t>(offsets.begin(), offsets.end());
  auto space_id_segment_iter = thrust::make_zip_iterator(space_id_iter, segments.begin());

  thrust::sort_by_key(rmm::exec_policy(stream),
                      space_id_segment_iter,
                      space_id_segment_iter + segments.size(),
                      segments.begin(),
                      segment_comparator<index_t, T>{});

  auto [threads_per_block, num_blocks] = grid_1d(num_spaces);
  simple_find_and_combine_segments_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    offsets, segments, merged_flag);

  CUSPATIAL_CHECK_CUDA(stream.value());
}

}  // namespace detail
}  // namespace cuspatial
