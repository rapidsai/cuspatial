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

#include <rmm/cuda_stream_view.hpp>

namespace cuspatial {

/**
 * @brief
 *
 * @tparam Cart2dItA
 * @tparam Cart2dItB
 * @tparam OffsetIteratorA
 * @tparam OffsetIteratorB
 * @tparam OffsetIteratorC
 * @tparam OutputIt
 * @param points_geometry_offsets_first
 * @param points_geometry_offsets_last
 * @param points_first
 * @param points_last
 * @param linestring_geometry_offsets_first
 * @param linestring_part_offsets_first
 * @param linestring_part_offsets_last
 * @param linestring_points_first
 * @param linestring_points_last
 * @param output_first Output iterator to a 3-tuple array. The first element should be compatible
 * with iterator_value_type<OffsetIteratorB>, stores the geometry index of the neareast linestring.
 * The second element should be compatible with iterator_value_type<OffsetIteratorC>, stores the
 * part index of the nearest segment. The third element should be compatible with vec_2d, stores the
 * nearest point.
 * @param stream
 * @return
 */
template <class Vec2dItA,
          class Vec2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
OutputIt pairwise_point_linestring_nearest_point(
  OffsetIteratorA points_geometry_offsets_first,
  OffsetIteratorA points_geometry_offsets_last,
  Vec2dItA points_first,
  Vec2dItA points_last,
  OffsetIteratorB linestring_geometry_offsets_first,
  OffsetIteratorC linestring_part_offsets_first,
  OffsetIteratorC linestring_part_offsets_last,
  Vec2dItB linestring_points_first,
  Vec2dItB linestring_points_last,
  OutputIt output_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);
}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_linestring_nearest_point.cuh>
