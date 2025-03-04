/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include <cuspatial/detail/range/enumerate_range.cuh>
#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/optional>
#include <thrust/pair.h>

namespace cuspatial {

/**
 * @addtogroup ranges
 */

/**
 * @brief Non-owning range-based interface to multilinestring data
 *
 * Provides a range-based interface to contiguous storage of multilinestring data, to make it easier
 * to access and iterate over multilinestrings, linestrings and points.
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
 * The underlying iterator must be device accessible if used in device kernel.
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

  CUSPATIAL_HOST_DEVICE multilinestring_range(GeometryIterator geometry_begin,
                                              GeometryIterator geometry_end,
                                              PartIterator part_begin,
                                              PartIterator part_end,
                                              VecIterator points_begin,
                                              VecIterator points_end);

  /// Return the number of multilinestrings in the array.
  CUSPATIAL_HOST_DEVICE auto size() { return num_multilinestrings(); }

  /// Return the number of multilinestrings in the array.
  CUSPATIAL_HOST_DEVICE auto num_multilinestrings();

  /// Return the total number of linestrings in the array.
  CUSPATIAL_HOST_DEVICE auto num_linestrings();

  /// Return the total number of points in the array.
  CUSPATIAL_HOST_DEVICE auto num_points();

  /// Return the iterator to the first multilinestring in the range.
  CUSPATIAL_HOST_DEVICE auto multilinestring_begin();

  /// Return the iterator to the one past the last multilinestring in the range.
  CUSPATIAL_HOST_DEVICE auto multilinestring_end();

  /// Return the iterator to the first multilinestring in the range.
  CUSPATIAL_HOST_DEVICE auto begin() { return multilinestring_begin(); }

  /// Return the iterator to the one past the last multilinestring in the range.
  CUSPATIAL_HOST_DEVICE auto end() { return multilinestring_end(); }

  /// Return the iterator to the first point in the range.
  CUSPATIAL_HOST_DEVICE auto point_begin() { return _point_begin; }

  /// Return the iterator to the one past the last point in the range.
  CUSPATIAL_HOST_DEVICE auto point_end() { return _point_end; }

  /// Return the iterator to the first geometry offset in the range.
  CUSPATIAL_HOST_DEVICE auto geometry_offset_begin() { return _geometry_begin; }

  /// Return the iterator to the one past the last geometry offset in the range.
  CUSPATIAL_HOST_DEVICE auto geometry_offset_end() { return _geometry_end; }

  /// Return the iterator to the first part offset in the range.
  CUSPATIAL_HOST_DEVICE auto part_offset_begin() { return _part_begin; }

  /// Return the iterator to the one past the last part offset in the range.
  CUSPATIAL_HOST_DEVICE auto part_offset_end() { return _part_end; }

  /// Given the index of a point, return the part (linestring) index where the point locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto part_idx_from_point_idx(IndexType point_idx);

  /// Given the index of a segment, return the part (linestring) index where the segment locates.
  /// If the segment id is invalid, returns nullopt.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE
    cuda::std::optional<typename thrust::iterator_traits<PartIterator>::difference_type>
    part_idx_from_segment_idx(IndexType point_idx);

  /// Given the index of a part (linestring), return the geometry (multilinestring) index
  /// where the linestring locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_part_idx(IndexType part_idx);

  /// Given the index of a point, return the geometry (multilinestring) index where the
  /// point locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto geometry_idx_from_point_idx(IndexType point_idx);

  // Given index to a linestring, return the index of the linestring inside its multilinestring.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto intra_part_idx(IndexType global_part_idx);

  // Given index to a point, return the index of the point inside its linestring.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto intra_point_idx(IndexType global_point_idx);

  /// Given an index of a segment, returns true if the index is valid.
  /// The index of a segment is the same as the index to the starting point of the segment.
  /// Thus, the index to the last point of a linestring is an invalid segment index.
  template <typename IndexType1, typename IndexType2>
  CUSPATIAL_HOST_DEVICE bool is_valid_segment_id(IndexType1 segment_idx, IndexType2 part_idx);

  /// Returns the segment given a segment index.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto segment(IndexType segment_idx);

  /// Returns an iterator to the counts of points per multilinestring
  CUSPATIAL_HOST_DEVICE auto multilinestring_point_count_begin();

  /// Returns an iterator to the counts of segments per multilinestring
  CUSPATIAL_HOST_DEVICE auto multilinestring_point_count_end();

  /// Returns an iterator to the counts of points per multilinestring
  CUSPATIAL_HOST_DEVICE auto multilinestring_linestring_count_begin();

  /// Returns an iterator to the counts of points per multilinestring
  CUSPATIAL_HOST_DEVICE auto multilinestring_linestring_count_end();

  /// @internal
  /// Returns the owning class that provides views into the segments of the multilinestring range
  /// Can only be constructed on host
  auto _segments(rmm::cuda_stream_view);

  /// Returns the `multilinestring_idx`th multilinestring in the range.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto operator[](IndexType multilinestring_idx);

  /// Range Casts

  /// Casts the multilinestring range into a multipoint range.
  /// This treats each multilinestring as simply a collection of points,
  /// ignoring all edges in the multilinestring.
  CUSPATIAL_HOST_DEVICE auto as_multipoint_range();

 protected:
  GeometryIterator _geometry_begin;
  GeometryIterator _geometry_end;
  PartIterator _part_begin;
  PartIterator _part_end;
  VecIterator _point_begin;
  VecIterator _point_end;

 private:
  /// @internal
  /// Return the iterator to the part index where the point locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto _part_iter_from_point_idx(IndexType point_idx);
  /// @internal
  /// Return the iterator to the geometry index where the part locates.
  template <typename IndexType>
  CUSPATIAL_HOST_DEVICE auto _geometry_iter_from_part_idx(IndexType part_idx);
};

/**
 * @brief Create a multilinestring_range object from size and start iterators
 *
 * @tparam GeometryIteratorDiffType Index type of the size of the geometry array
 * @tparam PartIteratorDiffType Index type of the size of the part array
 * @tparam VecIteratorDiffType Index type of the size of the point array
 * @tparam GeometryIterator iterator type for offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam PartIterator iterator type for the part offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 * @tparam VecIterator iterator type for the point array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI].
 *
 * @note Iterators must be device-accessible if the view is intended to be
 * used on device.
 *
 * @param num_multilinestrings Number of multilinestrings in the array.
 * @param geometry_begin Iterator to the start of the geometry array.
 * @param num_linestrings Number of linestrings in the underlying parts array.
 * @param part_begin Iterator to the start of the part array.
 * @param num_points Number of points in the underlying points array.
 * @param point_begin Iterator to the start of the point array.
 * @return A `multilinestring_range` object
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename GeometryIteratorDiffType,
          typename PartIteratorDiffType,
          typename VecIteratorDiffType,
          typename GeometryIterator,
          typename PartIterator,
          typename VecIterator>
auto make_multilinestring_range(GeometryIteratorDiffType num_multilinestrings,
                                GeometryIterator geometry_begin,
                                PartIteratorDiffType num_linestrings,
                                PartIterator part_begin,
                                VecIteratorDiffType num_points,
                                VecIterator point_begin)
{
  return multilinestring_range{geometry_begin,
                               geometry_begin + num_multilinestrings + 1,
                               part_begin,
                               part_begin + num_linestrings + 1,
                               point_begin,
                               point_begin + num_points};
}

/**
 * @brief Create a range object of multilinestring data from offset and point ranges
 *
 * @tparam IntegerRange1 Range to integers
 * @tparam IntegerRange2 Range to integers
 * @tparam PointRange Range to points
 *
 * @param geometry_offsets Range to multilinestring geometry offsets
 * @param part_offsets Range to linestring part offsets
 * @param points Range to underlying points
 * @return A multilinestring_range object
 */
template <typename IntegerRange1, typename IntegerRange2, typename PointRange>
auto make_multilinestring_range(IntegerRange1 geometry_offsets,
                                IntegerRange2 part_offsets,
                                PointRange points)
{
  return multilinestring_range(geometry_offsets.begin(),
                               geometry_offsets.end(),
                               part_offsets.begin(),
                               part_offsets.end(),
                               points.begin(),
                               points.end());
}

/**
 * @brief Create a range object of multilinestring from cuspatial::geometry_column_view.
 * Specialization for linestrings column.
 *
 * @pre linestrings_column must be a cuspatial::geometry_column_view
 */
template <collection_type_id Type,
          typename T,
          typename IndexType,
          typename GeometryColumnView,
          CUSPATIAL_ENABLE_IF(Type == collection_type_id::SINGLE)>
auto make_multilinestring_range(GeometryColumnView const& linestrings_column)
{
  CUSPATIAL_EXPECTS(linestrings_column.geometry_type() == geometry_type_id::LINESTRING,
                    "Must be Linestring geometry type.");
  auto geometry_iter       = thrust::make_counting_iterator(0);
  auto const& part_offsets = linestrings_column.offsets();
  auto const& points_xy = linestrings_column.child().child(1);  // Ignores x-y offset {0, 2, 4...}

  auto points_it = make_vec_2d_iterator(points_xy.template begin<T>());

  return multilinestring_range(geometry_iter,
                               geometry_iter + part_offsets.size(),
                               part_offsets.template begin<IndexType>(),
                               part_offsets.template end<IndexType>(),
                               points_it,
                               points_it + points_xy.size() / 2);
}

/**
 * @brief Create a range object of multilinestring from cuspatial::geometry_column_view.
 * Specialization for multilinestrings column.
 *
 * @pre linestring_column must be a cuspatial::geometry_column_view
 */
template <collection_type_id Type,
          typename T,
          typename IndexType,
          CUSPATIAL_ENABLE_IF(Type == collection_type_id::MULTI),
          typename GeometryColumnView>
auto make_multilinestring_range(GeometryColumnView const& linestrings_column)
{
  CUSPATIAL_EXPECTS(linestrings_column.geometry_type() == geometry_type_id::LINESTRING,
                    "Must be Linestring geometry type.");
  auto const& geometry_offsets = linestrings_column.offsets();
  auto const& parts            = linestrings_column.child();
  auto const& part_offsets     = parts.child(0);
  auto const& points_xy        = parts.child(1).child(1);  // Ignores x-y offset {0, 2, 4...}

  auto points_it = make_vec_2d_iterator(points_xy.template begin<T>());

  return multilinestring_range(geometry_offsets.template begin<IndexType>(),
                               geometry_offsets.template end<IndexType>(),
                               part_offsets.template begin<IndexType>(),
                               part_offsets.template end<IndexType>(),
                               points_it,
                               points_it + points_xy.size() / 2);
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include <cuspatial/detail/range/multilinestring_range.cuh>
