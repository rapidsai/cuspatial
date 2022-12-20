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

#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/pair.h>

namespace cuspatial {

enum IntersectionTypeCode : uint8_t { POINT = 0, LINESTRING = 1 };

/**
 * @brief Result of linestring intersections
 *
 * Owning object to the result of linestring intersections.
 * The results are modeled after arrow type List<Union<Point, LineString>>.
 *
 * @tparam T Type of coordinates
 * @tparam OffsetType Type of offsets
 */
template <typename T, typename OffsetType>
struct linestring_intersection_result {
  using point_t   = vec_2d<T>;
  using segment_t = segment<T>;
  using types_t   = uint8_t;
  using index_t   = OffsetType;

  /// List offsets to the union column
  std::unique_ptr<rmm::device_uvector<index_t>> geometry_collection_offset;

  /// Union Column Results
  std::unique_ptr<rmm::device_uvector<types_t>> types_buffer;
  std::unique_ptr<rmm::device_uvector<index_t>> offset_buffer;

  /// Child 0: Point Results
  std::unique_ptr<rmm::device_uvector<point_t>> points_coords;

  /// Child 1: Segment Results
  std::unique_ptr<rmm::device_uvector<segment_t>> segments_coords;

  /// Look-back Indices
  std::unique_ptr<rmm::device_uvector<index_t>> lhs_linestring_id;
  std::unique_ptr<rmm::device_uvector<index_t>> lhs_segment_id;
  std::unique_ptr<rmm::device_uvector<index_t>> rhs_linestring_id;
  std::unique_ptr<rmm::device_uvector<index_t>> rhs_segment_id;
};

/**
 * @brief Compute the intersections between multilinestrings and ids to the intersecting
 * linestrings.
 *
 * @tparam T Type of coordinate
 * @tparam index_t Type of the look-back index in result
 * @tparam MultiLinestringRange1 Multilinestring Range of the first operand
 * @tparam MultiLinestringRange2 Multilinestring Range of the second operand
 *
 * @param multilinestrings1 Range to the first multilinestring in the pair
 * @param multilinestrings2 Range to the second multilinestring in the pair
 * @param mr The resource to use to allocate the returned data
 * @param stream The CUDA stream on which to perform computations
 * @return Intersection Result
 */
template <typename T,
          typename index_t,
          typename MultiLinestringRange1,
          typename MultiLinestringRange2>
linestring_intersection_result<T, index_t> pairwise_linestring_intersection(
  MultiLinestringRange1 multilinestrings1,
  MultiLinestringRange2 multilinestrings2,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/linestring_intersection.cuh>
