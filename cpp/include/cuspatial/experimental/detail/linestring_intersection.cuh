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

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/detail/find/find_and_combine_segment.cuh>
#include <cuspatial/experimental/detail/find/find_duplicate_points.cuh>
#include <cuspatial/experimental/detail/find/find_points_on_segments.cuh>
#include <cuspatial/experimental/detail/linestring_intersection_count.cuh>
#include <cuspatial/experimental/detail/linestring_intersection_with_duplicates.cuh>
#include <cuspatial/experimental/ranges/multipoint_range.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>
#include <thrust/unique.h>

#include <cuda/atomic>

#include <cstdint>
#include <memory>

namespace cuspatial {
template <typename T, typename OffsetType>
struct linestring_intersection_result;

namespace detail {

/**
 * @brief Functor to gather the look-back ids from the ids in intermediates.
 *
 * In the final step, intersection function is required to compute a single look-back id
 * column for the union column rows. This functor looks at each row of the union
 * column, determine its geometry type, and gather the look-back id from the
 * corresponding id range.
 */
template <typename OffsetRange, typename TypeRange>
struct gather_ids_functor {
  id_ranges<OffsetRange> points;
  id_ranges<OffsetRange> segments;

  TypeRange types_buffer;
  OffsetRange offset_buffer;

  gather_ids_functor(id_ranges<OffsetRange> points,
                     id_ranges<OffsetRange> segments,
                     TypeRange types_buffer,
                     OffsetRange offset_buffer)
    : points(points), segments(segments), types_buffer(types_buffer), offset_buffer(offset_buffer)
  {
  }

  template <typename IndexType>
  auto __device__ operator()(IndexType i)
  {
    if (types_buffer[i] == IntersectionTypeCode::POINT) {
      return points[offset_buffer[i]];
    } else {
      return segments[offset_buffer[i]];
    }
  }
};

/**
 * @brief Functor to compute types buffer in the final union column result.
 */
template <typename type_t, typename OffsetRangeA, typename OffsetRangeB>
struct types_buffer_functor {
  OffsetRangeA geometric_column_offset;
  OffsetRangeB points_offset;
  OffsetRangeB segments_offset;

  types_buffer_functor(OffsetRangeA geometric_column_offset,
                       OffsetRangeB points_offset,
                       OffsetRangeB segments_offset)
    : geometric_column_offset(geometric_column_offset),
      points_offset(points_offset),
      segments_offset(segments_offset)
  {
  }

  template <typename index_t>
  type_t __device__ operator()(index_t i)
  {
    auto geometry_idx = thrust::distance(
      geometric_column_offset.begin(),
      thrust::prev(thrust::upper_bound(
        thrust::seq, geometric_column_offset.begin(), geometric_column_offset.end(), i)));

    auto num_points   = points_offset[geometry_idx + 1] - points_offset[geometry_idx];
    auto num_segments = segments_offset[geometry_idx + 1] - segments_offset[geometry_idx];
    auto pair_offset  = geometric_column_offset[geometry_idx];

    // In each pair, points always precedes segment (arbitrarily).
    if (pair_offset <= i && i < pair_offset + num_points)
      return IntersectionTypeCode::POINT;
    else
      return IntersectionTypeCode::LINESTRING;
  }
};

/**
 * @brief Compute types buffer in the final union column result.
 */
template <typename types_t, typename index_t, typename OffsetRangeA, typename OffsetRangeB>
std::unique_ptr<rmm::device_uvector<types_t>> compute_types_buffer(
  index_t union_column_size,
  OffsetRangeA geometric_column_offset,
  OffsetRangeB points_offset,
  OffsetRangeB segments_offset,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto types_buffer = std::make_unique<rmm::device_uvector<types_t>>(union_column_size, stream, mr);
  thrust::tabulate(rmm::exec_policy(stream),
                   types_buffer->begin(),
                   types_buffer->end(),
                   types_buffer_functor<types_t, OffsetRangeA, OffsetRangeB>(
                     geometric_column_offset, points_offset, segments_offset));
  return types_buffer;
}

/**
 * @brief Compute union column's offset buffer
 *
 * This is performing a group-by cummulative sum (pandas semantic) operation
 * to an "all 1s vector", using `types_buffer` as the key column.
 */
template <typename index_t>
std::unique_ptr<rmm::device_uvector<index_t>> compute_offset_buffer(
  rmm::device_uvector<uint8_t> const& types_buffer,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  auto N            = types_buffer.size();
  auto keys_copy    = rmm::device_uvector(types_buffer, stream);
  auto indices_temp = rmm::device_uvector<index_t>(N, stream);
  thrust::sequence(rmm::exec_policy(stream), indices_temp.begin(), indices_temp.end());
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream), keys_copy.begin(), keys_copy.end(), indices_temp.begin());

  auto offset_buffer = std::make_unique<rmm::device_uvector<index_t>>(N, stream, mr);
  thrust::uninitialized_fill_n(rmm::exec_policy(stream), offset_buffer->begin(), N, 1);
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                keys_copy.begin(),
                                keys_copy.end(),
                                offset_buffer->begin(),
                                offset_buffer->begin());
  thrust::scatter(rmm::exec_policy(stream),
                  offset_buffer->begin(),
                  offset_buffer->end(),
                  indices_temp.begin(),
                  offset_buffer->begin());
  return offset_buffer;
}

}  // namespace detail

/**
 * @brief Compute intersections between multilnestrings with duplicates.
 */
template <typename T,
          typename index_t,
          typename MultiLinestringRange1,
          typename MultiLinestringRange2>
linestring_intersection_result<T, index_t> pairwise_linestring_intersection(
  MultiLinestringRange1 multilinestrings1,
  MultiLinestringRange2 multilinestrings2,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  using types_t = typename linestring_intersection_result<T, index_t>::types_t;

  static_assert(is_same_floating_point<T, typename MultiLinestringRange2::element_t>(),
                "Inputs and output must have the same floating point value type.");

  static_assert(is_same<vec_2d<T>,
                        typename MultiLinestringRange1::point_t,
                        typename MultiLinestringRange2::point_t>(),
                "All input types must be cuspatial::vec_2d with the same value type");

  CUSPATIAL_EXPECTS(multilinestrings1.size() == multilinestrings2.size(),
                    "The size input multilinestrings mismatch.");

  auto const num_pairs = multilinestrings1.size();

  // Phase 1 and 2: Estimate and compute duplicates
  auto [points, segments] = detail::pairwise_linestring_intersection_with_duplicates<index_t, T>(
    multilinestrings1, multilinestrings2, mr, stream);
  auto num_points   = points.num_geoms();
  auto num_segments = segments.num_geoms();

  // Phase 3: Remove duplicate points from intermediates
  // TODO: improve memory usage by using IIFE to
  // Remove the duplicate points
  rmm::device_uvector<int32_t> point_flags(num_points, stream);
  detail::find_duplicate_points(
    make_multipoint_range(points.offset_range(), points.geom_range()), point_flags.begin(), stream);

  points.remove_if(range(point_flags.begin(), point_flags.end()), stream);
  point_flags.resize(points.geoms->size(), stream);

  // Merge mergable segments
  rmm::device_uvector<uint8_t> segment_flags(num_segments, stream);
  detail::find_and_combine_segment(
    segments.offset_range(), segments.geom_range(), segment_flags.begin(), stream);
  segments.remove_if(range(segment_flags.begin(), segment_flags.end()), stream);

  // Merge point on segments
  detail::find_points_on_segments(make_multipoint_range(points.offset_range(), points.geom_range()),
                                  segments.offset_range(),
                                  segments.geom_range(),
                                  point_flags.begin(),
                                  stream);

  points.remove_if(range(point_flags.begin(), point_flags.end()), stream);

  // Phase 4: Assemble results as union column
  auto num_union_column_rows = points.geoms->size() + segments.geoms->size();
  auto geometry_collection_offsets =
    std::make_unique<rmm::device_uvector<index_t>>(num_pairs + 1, stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    points.offsets->begin(),
                    points.offsets->end(),
                    segments.offsets->begin(),
                    geometry_collection_offsets->begin(),
                    thrust::plus<index_t>());

  auto types_buffer = detail::compute_types_buffer<types_t>(
    num_union_column_rows,
    range(geometry_collection_offsets->begin(), geometry_collection_offsets->end()),
    points.offset_range(),
    segments.offset_range(),
    stream,
    mr);

  auto offsets_buffer = detail::compute_offset_buffer<index_t>(*types_buffer, mr, stream);

  // Assemble the look-back ids.
  auto lhs_linestring_id =
    std::make_unique<rmm::device_uvector<index_t>>(num_union_column_rows, stream, mr);
  auto lhs_segment_id =
    std::make_unique<rmm::device_uvector<index_t>>(num_union_column_rows, stream, mr);
  auto rhs_linestring_id =
    std::make_unique<rmm::device_uvector<index_t>>(num_union_column_rows, stream, mr);
  auto rhs_segment_id =
    std::make_unique<rmm::device_uvector<index_t>>(num_union_column_rows, stream, mr);

  auto id_iter = thrust::make_zip_iterator(lhs_linestring_id->begin(),
                                           lhs_segment_id->begin(),
                                           rhs_linestring_id->begin(),
                                           rhs_segment_id->begin());
  thrust::tabulate(rmm::exec_policy(stream),
                   id_iter,
                   id_iter + num_union_column_rows,
                   gather_ids_functor{points.get_id_ranges(),
                                      segments.get_id_ranges(),
                                      range(types_buffer->begin(), types_buffer->end()),
                                      range(offsets_buffer->begin(), offsets_buffer->end())});

  return linestring_intersection_result<T, index_t>{std::move(geometry_collection_offsets),
                                                    std::move(types_buffer),
                                                    std::move(offsets_buffer),
                                                    std::move(points.geoms),
                                                    std::move(segments.geoms),
                                                    std::move(lhs_linestring_id),
                                                    std::move(lhs_segment_id),
                                                    std::move(rhs_linestring_id),
                                                    std::move(rhs_segment_id)};
}

}  // namespace cuspatial
