/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial/types.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <initializer_list>
#include <memory>

namespace cuspatial {
namespace test {

using namespace cudf;
using namespace cudf::test;

std::unique_ptr<column> coords_offsets(size_type num_points, rmm::cuda_stream_view stream)
{
  auto zero = make_fixed_width_scalar<size_type>(0, stream);
  auto two  = make_fixed_width_scalar<size_type>(2, stream);

  return sequence(num_points + 1, *zero, *two);
}

template <typename T>
std::unique_ptr<column> make_non_nullable_lists_column(std::unique_ptr<column> offset,
                                                       std::unique_ptr<column> child)
{
  auto size = offset->size() - 1;
  return make_lists_column(size, std::move(offset), std::move(child), 0, {});
}

template <typename T>
std::unique_ptr<column> make_non_nullable_lists_column(std::initializer_list<size_type> offset,
                                                       std::unique_ptr<column> child)
{
  auto d_offset = fixed_width_column_wrapper<size_type>(offset).release();
  return make_non_nullable_lists_column<T>(std::move(d_offset), std::move(child));
}

template <typename T>
std::unique_ptr<column> make_non_nullable_lists_column(std::unique_ptr<column> offset,
                                                       std::initializer_list<T> child)
{
  auto d_child = fixed_width_column_wrapper<T>(child).release();
  return make_non_nullable_lists_column<T>(std::move(offset), std::move(d_child));
}

/**
 * @brief helper function to make a point column
 *
 * A point column has cudf type LIST<FLOAT | DOUBLE>
 *
 * Example:
 * [POINT (0 0), POINT (1 1), POINT (2 2)]
 * Offset 0 2 4 6
 * Child  0 0 1 1 2 2
 *
 * @tparam T Coordinate value type
 * @param point_coords interleaved x-y coordinates of the points
 * @param stream The CUDA stream on which to perform computations
 *
 * @return Intersection Result
 * @return A cudf LIST column with point data
 */
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_point_column(
  std::initializer_list<T>&& point_coords, rmm::cuda_stream_view stream)
{
  auto num_points = point_coords.size() / 2;

  return {collection_type_id::SINGLE,
          make_non_nullable_lists_column<T>(coords_offsets(num_points, stream), point_coords)};
}

/**
 * @brief helper function to make a multipoint column
 *
 * A multipoint column has cudf type LIST<LIST<FLOAT | DOUBLE>>
 *
 * Example:
 * [MULTIPOINT (POINT (0 0), POINT (1 1)), MULTIPOINT (POINT (2 2), POINT (3 3))]
 * Offset 0 2 5
 * Offset 0 2 4 6 8
 * Child  0 0 1 1 2 2 3 3
 *
 * @tparam T Coordinate value type
 * @param multipoint_offsets Offset to the starting position for each multipoint
 * @param point_coords Interleaved x-y coordinates of the points
 * @param stream The CUDA stream on which to perform computations
 *
 * @return A cudf LIST column with multipoint data
 */
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_point_column(
  std::initializer_list<cudf::size_type>&& multipoint_offsets,
  std::initializer_list<T> point_coords,
  rmm::cuda_stream_view stream)
{
  auto num_points = point_coords.size() / 2;

  return {collection_type_id::MULTI,
          make_non_nullable_lists_column<T>(
            multipoint_offsets,
            make_non_nullable_lists_column<T>(coords_offsets(num_points, stream), point_coords))};
}

/**
 * @brief helper function to make a linestring column
 *
 * A linestring column has cudf type LIST<LIST<FLOAT | DOUBLE>>
 *
 * Example:
 * [LINESTRING (0 0, 1 1, 2 2), LINESTRING (3 3, 4 4)]
 * Offset 0 3 5
 * Offset 0 2 4 6 8
 * Child  0 0 1 1 2 2 3 3 4 4
 *
 * @tparam T Coordinate value type
 * @param linestring_offsets Offset to the starting position for each linestring
 * @param point_coords Interleaved x-y coordinates of the points
 * @param stream The CUDA stream on which to perform computations
 *
 * @return A cudf LIST column with linestring data
 */
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_linestring_column(
  std::initializer_list<cudf::size_type>&& linestring_offsets,
  std::initializer_list<T>&& linestring_coords,
  rmm::cuda_stream_view stream)
{
  auto num_points = linestring_coords.size() / 2;

  return {
    collection_type_id::SINGLE,
    make_non_nullable_lists_column<T>(
      linestring_offsets,
      make_non_nullable_lists_column<T>(coords_offsets(num_points, stream), linestring_coords))};
}

/**
 * @brief helper function to make a multilinestring column
 *
 * A multilinestring column has cudf type LIST<LIST<LIST<FLOAT | DOUBLE>>>
 *
 * Example:
 * [
 *    MULTILINESTRING (LINESTRING (0 0, 1 1), LINESTRING (2 2, 3 3)),
 *    MULTILINESTRING (LINESTRING (4 4, 5 5))
 * ]
 * Offset 0 2 3
 * Offset 0 2 4 6
 * Offset 0 2 4 6 8 10
 * Child  0 0 1 1 2 2 3 3 4 4 5 5
 *
 * @tparam T Coordinate value type
 * @param multilinestring_offsets Offset to the starting position for each multilinestring
 * @param linestring_offsets Offset to the starting position for each linestring
 * @param point_coords Interleaved x-y coordinates of the points
 * @param stream The CUDA stream on which to perform computations
 *
 * @return A cudf LIST column with multilinestring data
 */
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_linestring_column(
  std::initializer_list<cudf::size_type>&& multilinestring_offsets,
  std::initializer_list<cudf::size_type>&& linestring_offsets,
  std::initializer_list<T> linestring_coords,
  rmm::cuda_stream_view stream)
{
  auto num_points = linestring_coords.size() / 2;
  return {
    collection_type_id::MULTI,
    make_non_nullable_lists_column<T>(
      multilinestring_offsets,
      make_non_nullable_lists_column<T>(
        linestring_offsets,
        make_non_nullable_lists_column<T>(coords_offsets(num_points, stream), linestring_coords)))};
}

/**
 * @brief helper function to make a polygon column
 *
 * A polygon column has cudf type LIST<LIST<LIST<FLOAT | DOUBLE>>>
 *
 * Example:
 * [
 *    POLYGON ((0 0, 1 1, 0 1, 0 0), (0 0, -1 0, -1 -1, 0 0)),
 *    POLYGON ((3 3, 4 4, 3 4, 3 3))
 * ]
 * Offset 0 2 3
 * Offset 0 4 8 12
 * Offset 0 2 4 6 8 10 12 14 16 18 20 22 24
 * Child  0 0 1 1 0 1 0 0 0 0 -1 0 -1 -1 0 0 3 3 4 4 3 4 3 3
 *
 * @tparam T Coordinate value type
 * @param polygon_offsets Offset to the starting position for each polygon
 * @param ring_offsets Offset to the starting position for each ring
 * @param point_coords Interleaved x-y coordinates of the points
 * @param stream The CUDA stream on which to perform computations
 *
 * @return A cudf LIST column with polygon data
 */
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_polygon_column(
  std::initializer_list<cudf::size_type>&& polygon_offsets,
  std::initializer_list<cudf::size_type>&& ring_offsets,
  std::initializer_list<T> polygon_coords,
  rmm::cuda_stream_view stream)
{
  auto num_points = polygon_coords.size() / 2;
  return {
    collection_type_id::SINGLE,
    make_non_nullable_lists_column<T>(
      polygon_offsets,
      make_non_nullable_lists_column<T>(
        ring_offsets,
        make_non_nullable_lists_column<T>(coords_offsets(num_points, stream), polygon_coords)))};
}

/**
 * @brief helper function to make a multipolygon column
 *
 * A multipolygon column has cudf type LIST<LIST<LIST<LIST<FLOAT | DOUBLE>>>>
 *
 * Example:
 * [
 *    MULTIPOLYGON (POLYGON (0 0, 1 1, 0 1, 0 0), POLYGON (0 0, -1 0, -1 -1, 0 0)),
 *    MULTIPOLYGON (POLYGON ((3 3, 4 4, 3 4, 3 3))
 * ]
 *
 * Offset 0 1 2 3
 * Offset 0 4 8 12
 * Offset 0 2 4 6 8 10 12 14 16 18 20 22 24
 * Child  0 0 1 1 0 1 0 0 0 0 -1 0 -1 -1 0 0 3 3 4 4 3 4 3 3
 *
 * @tparam T Coordinate value type
 * @param multipolygon_offsets Offset to the starting position for each multipolygon
 * @param polygon_offsets Offset to the starting position for each polygon
 * @param ring_offsets Offset to the starting position for each ring
 * @param point_coords Interleaved x-y coordinates of the points
 * @param stream The CUDA stream on which to perform computations
 *
 * @return A cudf LIST column with multipolygon data
 */
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_polygon_column(
  std::initializer_list<cudf::size_type>&& multipolygon_offsets,
  std::initializer_list<cudf::size_type>&& polygon_offsets,
  std::initializer_list<cudf::size_type>&& ring_offsets,
  std::initializer_list<T> polygon_coords,
  rmm::cuda_stream_view stream)
{
  auto num_points = polygon_coords.size() / 2;
  return {
    collection_type_id::MULTI,
    make_non_nullable_lists_column<T>(
      multipolygon_offsets,
      make_non_nullable_lists_column<T>(
        polygon_offsets,
        make_non_nullable_lists_column<T>(
          ring_offsets,
          make_non_nullable_lists_column<T>(coords_offsets(num_points, stream), polygon_coords))))};
}

}  // namespace test
}  // namespace cuspatial
