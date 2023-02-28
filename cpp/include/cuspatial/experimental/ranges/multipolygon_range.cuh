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
#include <cuspatial/experimental/detail/ranges/enumerate_range.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

namespace cuspatial {

/**
 * @brief Non-owning range-based interface to multipolygon data
 * @ingroup ranges
 *
 * Provides a range-based interface to contiguous storage of multipolygon data, to make it easier
 * to access and iterate over multipolygons, polygons, rings and points.
 *
 * Conforms to GeoArrow's specification of multipolygon:
 * https://github.com/geopandas/geo-arrow-spec/blob/main/format.md
 *
 * @tparam GeometryIterator iterator type for the geometry offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam PartIterator iterator type for the part offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam RingIterator iterator type for the ring offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam VecIterator iterator type for the point array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 *
 * @note Though this object is host/device compatible,
 * The underlying iterator must be device accessible if used in device kernel.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename GeometryIterator,
          typename PartIterator,
          typename RingIterator,
          typename VecIterator>
class multipolygon_range {
 public:
  using geometry_it_t = GeometryIterator;
  using part_it_t     = PartIterator;
  using ring_it_t     = RingIterator;
  using point_it_t    = VecIterator;
  using point_t       = iterator_value_type<VecIterator>;
  using element_t     = iterator_vec_base_type<VecIterator>;

  multipolygon_range(GeometryIterator geometry_begin,
                     GeometryIterator geometry_end,
                     PartIterator part_begin,
                     PartIterator part_end,
                     RingIterator ring_begin,
                     RingIterator ring_end,
                     VecIterator points_begin,
                     VecIterator points_end);

  /// Return the number of multipolygons in the array.
  CUSPATIAL_HOST_DEVICE auto size() { return num_multipolygons(); }

  /// Return the number of multipolygons in the array.
  CUSPATIAL_HOST_DEVICE auto num_multipolygons();

  /// Return the total number of polygons in the array.
  CUSPATIAL_HOST_DEVICE auto num_polygons();

  /// Return the total number of rings in the array.
  CUSPATIAL_HOST_DEVICE auto num_rings();

  /// Return the total number of points in the array.
  CUSPATIAL_HOST_DEVICE auto num_points();

  /// Return the iterator to the first multipolygon in the range.
  CUSPATIAL_HOST_DEVICE auto multipolygon_begin();

  /// Return the iterator to the one past the last multipolygon in the range.
  CUSPATIAL_HOST_DEVICE auto multipolygon_end();

  /// Return the iterator to the first multipolygon in the range.
  CUSPATIAL_HOST_DEVICE auto begin() { return multipolygon_begin(); }

  /// Return the iterator to the one past the last multipolygon in the range.
  CUSPATIAL_HOST_DEVICE auto end() { return multipolygon_end(); }

  /// Given the index of a point, return the ring index where the point locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto ring_idx_from_point_idx(IndexType point_idx);

  /// Given the index of a ring, return the part (polygon) index
  /// where the polygon locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto part_idx_from_ring_idx(IndexType ring_idx);

  /// Given the index of a part (polygon), return the geometry (multipolygon) index
  /// where the polygon locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_part_idx(IndexType part_idx);

  /// Given the index of a point, return the geometry (multipolygon) index where the
  /// point locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_point_idx(IndexType point_idx);

  /// Returns the `multipolygon_idx`th multipolygon in the range.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto operator[](IndexType multipolygon_idx);

 protected:
  GeometryIterator _geometry_begin;
  GeometryIterator _geometry_end;
  PartIterator _part_begin;
  PartIterator _part_end;
  VecIterator _point_begin;
  VecIterator _point_end;
};

}  // namespace cuspatial

#include <cuspatial/experimental/detail/ranges/multipolygon_range.cuh>
