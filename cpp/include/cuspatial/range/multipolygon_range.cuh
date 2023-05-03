/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cuspatial/detail/range/enumerate_range.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/types.hpp>

namespace cuspatial {

/**
 * @addtogroup ranges
 * @{
 */

/**
 * @brief Non-owning range-based interface to multipolygon data
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

  using index_t   = iterator_value_type<GeometryIterator>;
  using element_t = iterator_vec_base_type<VecIterator>;

  int64_t static constexpr INVALID_INDEX = -1;

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

  /// Return the total number of segments in the array.
  CUSPATIAL_HOST_DEVICE auto num_segments();

  /// Return the iterator to the first multipolygon in the range.
  CUSPATIAL_HOST_DEVICE auto multipolygon_begin();

  /// Return the iterator to the one past the last multipolygon in the range.
  CUSPATIAL_HOST_DEVICE auto multipolygon_end();

  /// Return the iterator to the first multipolygon in the range.
  CUSPATIAL_HOST_DEVICE auto begin() { return multipolygon_begin(); }

  /// Return the iterator to the one past the last multipolygon in the range.
  CUSPATIAL_HOST_DEVICE auto end() { return multipolygon_end(); }

  /// Return the iterator to the first point in the range.
  CUSPATIAL_HOST_DEVICE auto point_begin();

  /// Return the iterator to the one past the last point in the range.
  CUSPATIAL_HOST_DEVICE auto point_end();

  /// Return the iterator to the first part offset in the range.
  CUSPATIAL_HOST_DEVICE auto part_offset_begin() { return _part_begin; }

  /// Return the iterator to the one past the last part offset in the range.
  CUSPATIAL_HOST_DEVICE auto part_offset_end() { return _part_end; }

  /// Return the iterator to the first ring offset in the range.
  CUSPATIAL_HOST_DEVICE auto ring_offset_begin() { return _ring_begin; }

  /// Return the iterator to the one past the last ring offset in the range.
  CUSPATIAL_HOST_DEVICE auto ring_offset_end() { return _ring_end; }

  /// Given the index of a segment, return the index of the geometry (multipolygon) that contains
  /// the segment. Segment index is the index to the starting point of the segment. If the index is
  /// the last point of the ring, then it is not a valid index. This function returns
  /// multipolygon_range::INVALID_INDEX if the index is invalid.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_segment_idx(IndexType segment_idx);

  /// Given the index of a point, return the index of the ring that contains the point.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto ring_idx_from_point_idx(IndexType point_idx);

  /// Given the index of a ring, return the index of the part (polygon) that contains the point.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto part_idx_from_ring_idx(IndexType ring_idx);

  /// Given the index of a part (polygon), return the index of the geometry (multipolygon) that
  /// contains the part.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_part_idx(IndexType part_idx);

  /// Returns the `multipolygon_idx`th multipolygon in the range.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto operator[](IndexType multipolygon_idx);

  /// Returns the `segment_idx`th segment in the multipolygon range.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto get_segment(IndexType segment_idx);

  /// Returns `true` if `point_idx`th point is the first point of `geometry_idx`th
  /// multipolygon
  template <typename IndexType1, typename IndexType2>
  CUSPATIAL_HOST_DEVICE bool is_first_point_of_multipolygon(IndexType1 point_idx,
                                                            IndexType2 geometry_idx);

  /// Returns an iterator to the number of points of the first multipolygon
  /// @note The count includes the duplicate first and last point of the ring.
  CUSPATIAL_HOST_DEVICE auto multipolygon_point_count_begin();
  /// Returns the one past the iterator to the number of points of the last multipolygon
  /// @note The count includes the duplicate first and last point of the ring.
  CUSPATIAL_HOST_DEVICE auto multipolygon_point_count_end();

  /// Returns an iterator to the number of rings of the first multipolygon
  CUSPATIAL_HOST_DEVICE auto multipolygon_ring_count_begin();
  /// Returns the one past the iterator to the number of rings of the last multipolygon
  CUSPATIAL_HOST_DEVICE auto multipolygon_ring_count_end();

  /// Returns an iterator to the number of segments of the first multipolygon
  CUSPATIAL_HOST_DEVICE auto multipolygon_segment_count_begin();
  /// Returns the one past the iterator to the number of segments of the last multipolygon
  CUSPATIAL_HOST_DEVICE auto multipolygon_segment_count_end();

  /// Returns an iterator to the start of the segment
  CUSPATIAL_HOST_DEVICE auto segment_begin();

  /// Returns an iterator to the end of the segment
  CUSPATIAL_HOST_DEVICE auto segment_end();

  /// Range Casting

  /// Cast the range of multipolygons as a range of multipoints, ignoring all edge connections and
  /// ring relationships.
  CUSPATIAL_HOST_DEVICE auto as_multipoint_range();

  /// Cast the range of multipolygons as a range of multilinestrings, ignoring ring relationships.
  CUSPATIAL_HOST_DEVICE auto as_multilinestring_range();

 protected:
  GeometryIterator _geometry_begin;
  GeometryIterator _geometry_end;
  PartIterator _part_begin;
  PartIterator _part_end;
  RingIterator _ring_begin;
  RingIterator _ring_end;
  VecIterator _point_begin;
  VecIterator _point_end;

  // TODO: find a better name
  CUSPATIAL_HOST_DEVICE auto subtracted_ring_begin();
  CUSPATIAL_HOST_DEVICE auto subtracted_ring_end();

 private:
  template <typename IndexType1, typename IndexType2>
  CUSPATIAL_HOST_DEVICE bool is_valid_segment_id(IndexType1 segment_idx, IndexType2 ring_idx);
};

/**
 * @brief Create a range object of multipolygon from cuspatial::geometry_column_view.
 * Specialization for polygons column.
 *
 * @pre polygons_column must be a cuspatial::geometry_column_view
 */
template <collection_type_id Type,
          typename T,
          typename IndexType,
          typename GeometryColumnView,
          CUSPATIAL_ENABLE_IF(Type == collection_type_id::SINGLE)>
auto make_multipolygon_range(GeometryColumnView const& polygons_column)
{
  CUSPATIAL_EXPECTS(polygons_column.geometry_type() == geometry_type_id::POLYGON,
                    "Must be polygon geometry type.");
  auto geometry_iter       = thrust::make_counting_iterator(0);
  auto const& part_offsets = polygons_column.offsets();
  auto const& ring_offsets = polygons_column.child().child(0);
  auto const& points_xy =
    polygons_column.child().child(1).child(1);  // Ignores x-y offset {0, 2, 4...}

  auto points_it = make_vec_2d_iterator(points_xy.template begin<T>());

  return multipolygon_range(geometry_iter,
                            geometry_iter + part_offsets.size(),
                            part_offsets.template begin<IndexType>(),
                            part_offsets.template end<IndexType>(),
                            ring_offsets.template begin<IndexType>(),
                            ring_offsets.template end<IndexType>(),
                            points_it,
                            points_it + points_xy.size() / 2);
}

/**
 * @brief Create a range object of multipolygon from cuspatial::geometry_column_view.
 * Specialization for multipolygons column.
 *
 * @pre polygon_column must be a cuspatial::geometry_column_view
 */
template <collection_type_id Type,
          typename T,
          typename IndexType,
          CUSPATIAL_ENABLE_IF(Type == collection_type_id::MULTI),
          typename GeometryColumnView>
auto make_multipolygon_range(GeometryColumnView const& polygons_column)
{
  CUSPATIAL_EXPECTS(polygons_column.geometry_type() == geometry_type_id::POLYGON,
                    "Must be polygon geometry type.");
  auto const& geometry_offsets = polygons_column.offsets();
  auto const& part_offsets     = polygons_column.child().child(0);
  auto const& ring_offsets     = polygons_column.child().child(1).child(0);
  auto const& points_xy =
    polygons_column.child().child(1).child(1).child(1);  // Ignores x-y offset {0, 2, 4...}

  auto points_it = make_vec_2d_iterator(points_xy.template begin<T>());

  return multipolygon_range(geometry_offsets.template begin<IndexType>(),
                            geometry_offsets.template end<IndexType>(),
                            part_offsets.template begin<IndexType>(),
                            part_offsets.template end<IndexType>(),
                            ring_offsets.template begin<IndexType>(),
                            ring_offsets.template end<IndexType>(),
                            points_it,
                            points_it + points_xy.size() / 2);
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include <cuspatial/detail/range/multipolygon_range.cuh>
