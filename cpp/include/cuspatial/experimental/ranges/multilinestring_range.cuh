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

#include <thrust/pair.h>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

namespace cuspatial {

/**
 * @brief Host-Device view object of a multilinestring array
 * @ingroup ranges
 *
 * Conforms to GeoArrow's specification of multilinestring:
 * https://github.com/geopandas/geo-arrow-spec/blob/main/format.md
 *
 * @tparam GeometryIterator iterator type for the geometry offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam PartIterator iterator type for the part offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam VecIterator iterator type for the point array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 *
 * @note Though this object is host/device compatible,
 * The underlying iterator should be device accessible if used in device kernel.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename GeometryIterator, typename PartIterator, typename VecIterator>
class multilinestring_range {
 public:
  using geometry_it_t = GeometryIterator;
  using part_it_t     = PartIterator;
  using point_it_t    = VecIterator;
  using point_t       = iterator_value_type<VecIterator>;
  using element_t     = iterator_vec_base_type<VecIterator>;

  multilinestring_range(GeometryIterator geometry_begin,
                        GeometryIterator geometry_end,
                        PartIterator part_begin,
                        PartIterator part_end,
                        VecIterator points_begin,
                        VecIterator points_end);

  /**
   * @brief Return the number of multilinestrings in the array.
   */
  CUSPATIAL_HOST_DEVICE auto size();

  /**
   * @brief Return the number of multilinestrings in the array.
   */
  CUSPATIAL_HOST_DEVICE auto num_multilinestrings();

  /**
   * @brief Return the total number of linestrings in the array.
   */
  CUSPATIAL_HOST_DEVICE auto num_linestrings();

  /**
   * @brief Return the total number of points in the array.
   */
  CUSPATIAL_HOST_DEVICE auto num_points();

  /**
   * @brief Given the index of a point, return the part (linestring) index where the point locates.
   */
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto part_idx_from_point_idx(IndexType point_idx);

  /**
   * @brief Given the index of a part (linestring), return the geometry (multilinestring) index
   * where the linestring locates.
   */
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_part_idx(IndexType part_idx);

  /**
   * @brief Given the index of the point, return the geometry (multilinestring) index where the
   * point locates.
   */
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_point_idx(IndexType point_idx);

  /**
   * @brief Given an index of a segment, returns true if the index is valid.
   * The index of a segment is the same as the index to the starting point of the segment.
   * Thus, the index to the last point of a linestring is an invalid index for segment.
   */
  template <typename IndexType1, typename IndexType2>
  CUSPATIAL_HOST_DEVICE bool is_valid_segment_id(IndexType1 segment_idx, IndexType2 part_idx);

  /**
   * @brief Returns the segment given a segment index.
   */
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE thrust::pair<vec_2d<element_t>, vec_2d<element_t>> segment(
    IndexType segment_idx);

 protected:
  GeometryIterator geometry_begin;
  GeometryIterator geometry_end;
  PartIterator part_begin;
  PartIterator part_end;
  VecIterator points_begin;
  VecIterator points_end;
};

/**
 * @brief Create a view of multilinestring array from array size and start iterators
 * @ingroup ranges
 *
 * @tparam IndexType1 Index type of the size of the geometry array
 * @tparam IndexType2 Index type of the size of the part array
 * @tparam IndexType3 Index type of the size of the point array
 * @tparam GeometryIterator iterator type for offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam PartIterator iterator type for the part offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam VecIterator iterator type for the point array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 *
 * @note Iterators should be device-accessible if the view is intended to be
 * used on device.
 *
 * @param num_multilinestrings Number of multilinestrings in the array.
 * @param geometry_begin Iterator to the start of the geometry array.
 * @param num_linestrings Number of linestrings in the underlying parts array.
 * @param part_begin Iterator to the start of the part array.
 * @param num_points Number of points in the underlying points array.
 * @param point_begin Iterator to the start of the point array.
 * @return View object to the multilinestring array.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename IndexType1,
          typename IndexType2,
          typename IndexType3,
          typename GeometryIterator,
          typename PartIterator,
          typename VecIterator>
auto make_multilinestring_range(IndexType1 num_multilinestrings,
                                GeometryIterator geometry_begin,
                                IndexType2 num_linestrings,
                                PartIterator part_begin,
                                IndexType3 num_points,
                                VecIterator point_begin)
{
  return multilinestring_range{geometry_begin,
                               geometry_begin + num_multilinestrings + 1,
                               part_begin,
                               part_begin + num_linestrings + 1,
                               point_begin,
                               point_begin + num_points};
}

}  // namespace cuspatial

#include <cuspatial/experimental/detail/ranges/multilinestring_range.cuh>
