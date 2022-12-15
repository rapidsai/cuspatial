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

#pragma once

#include <cuspatial_test/test_util.cuh>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>
#include <thrust/tuple.h>

namespace cuspatial {
namespace detail {

/**
 * @brief Functor to find if the given point is on any of the segments in the same pair
 */
template <typename MultiPointRange, typename OffsetsRange, typename SegmentsRange>
struct find_point_on_segment_functor {
  MultiPointRange multipoints;
  OffsetsRange segment_offsets;
  SegmentsRange segments;

  find_point_on_segment_functor(MultiPointRange multipoints,
                                OffsetsRange segment_offsets,
                                SegmentsRange segments)
    : multipoints(multipoints), segment_offsets(segment_offsets), segments(segments)
  {
  }

  template <typename IndexType>
  uint8_t __device__ operator()(IndexType i)
  {
    auto point        = thrust::raw_reference_cast(multipoints.point_begin()[i]);
    auto geometry_idx = multipoints.geometry_idx_from_point_idx(i);

    for (auto segment_idx = segment_offsets[geometry_idx];
         segment_idx < segment_offsets[geometry_idx + 1];
         segment_idx++) {
      auto const& segment = thrust::raw_reference_cast(segments[segment_idx]);
      if (is_point_on_segment(segment, point)) return true;
    }
    return false;
  }
};

/**
 * @brief Given a multipoint and a set of segments, for each point, if the point is
 * on any of the segments, set the `mergeable_flag` of the point to `1`.
 */
template <typename MultiPointRange,
          typename OffsetsRange,
          typename SegmentsRange,
          typename OutputIt1>
void find_points_on_segments(MultiPointRange multipoints,
                             OffsetsRange segment_offsets,
                             SegmentsRange segments,
                             OutputIt1 mergeable_flag,
                             rmm::cuda_stream_view stream)
{
  using index_t = typename MultiPointRange::index_t;

  CUSPATIAL_EXPECTS(multipoints.size() == segment_offsets.size() - 1,
                    "Input should contain the same number of pairs.");

  if (segments.size() == 0) return;

  thrust::tabulate(rmm::exec_policy(stream),
                   mergeable_flag,
                   mergeable_flag + multipoints.num_points(),
                   find_point_on_segment_functor{multipoints, segment_offsets, segments});
}

}  // namespace detail
}  // namespace cuspatial
