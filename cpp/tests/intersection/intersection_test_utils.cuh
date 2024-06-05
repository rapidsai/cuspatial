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

#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/intersection.cuh>

#include <rmm/resource_ref.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace cuspatial {
namespace test {

// Custom order for two segments
template <typename T>
bool CUSPATIAL_HOST_DEVICE operator<(segment<T> lhs, segment<T> rhs)
{
  return lhs.v1 < rhs.v1 || (lhs.v1 == rhs.v1 && lhs.v2 < rhs.v2);
}

/**
 * @brief Functor for segmented sorting a geometry array
 *
 * Using a label array and a geometry array as keys, this functor defines that
 * all keys with smaller labels should precede keys with larger labels; and that
 * the order with the same label should be determined by the natural order of the
 * geometries.
 *
 * Example:
 * Labels: {0, 0, 0, 1}
 * Points: {(0, 0), (5, 5), (1, 1), (3, 3)}
 * Result: {(0, 0), (1, 1), (5, 5), (3, 3)}
 */
template <typename KeyType, typename GeomType>
struct order_key_value_pairs {
  using key_value_t = thrust::tuple<KeyType, GeomType>;

  bool CUSPATIAL_HOST_DEVICE operator()(key_value_t lhs, key_value_t rhs)
  {
    return thrust::get<0>(lhs) < thrust::get<0>(rhs) ||
           (thrust::get<0>(lhs) == thrust::get<0>(rhs) &&
            thrust::get<1>(lhs) < thrust::get<1>(rhs));
  }
};

/**
 * @brief Perform sorting to the intersection result
 *
 * The result of intersection result is non-determinisitc. This algorithm sorts
 * the geometries of the same types and the same list and makes the result deterministic.
 *
 * The example below contains 2 rows and 4 geometries. The order of the first
 * and second point is non-deterministic.
 * [
 *  [Point(1.0, 1.5), Point(0.0, -0.3), Segment((0.0, 0.0), (1.0, 1.0))]
 *   ^                ^
 *  [Point(-3, -5)]
 * ]
 *
 * After sorting, the result is deterministic:
 * [
 *  [Point(0.0, -0.3), Point(1.0, 1.5), Segment((0.0, 0.0), (1.0, 1.0))]
 *   ^                 ^
 *  [Point(-3, -5)]
 * ]
 *
 * This function invalidates the input @p result and return a copy of sorted results.
 */
template <typename T, typename IndexType, typename type_t>
linestring_intersection_result<T, IndexType> segment_sort_intersection_result(
  linestring_intersection_result<T, IndexType>& result,
  rmm::device_async_resource_ref mr,
  rmm::cuda_stream_view stream)
{
  auto const num_points   = result.points_coords->size();
  auto const num_segments = result.segments_coords->size();
  auto const num_geoms    = num_points + num_segments;

  rmm::device_uvector<IndexType> scatter_map(num_geoms, stream);
  thrust::sequence(rmm::exec_policy(stream), scatter_map.begin(), scatter_map.end());

  // Compute keys for each row in the union column. Rows of the same list
  // are assigned the same label.
  rmm::device_uvector<IndexType> geometry_collection_keys(num_geoms, stream);
  auto geometry_collection_keys_begin = make_geometry_id_iterator<IndexType>(
    result.geometry_collection_offset->begin(), result.geometry_collection_offset->end());

  thrust::copy(rmm::exec_policy(stream),
               geometry_collection_keys_begin,
               geometry_collection_keys_begin + num_geoms,
               geometry_collection_keys.begin());

  // Perform "group-by" based on the list label and type of the row -
  // This makes the geometry of the same type and of the same list neighbor.

  // Make a copy of types buffer so that the sorting does not affect the original.
  auto types_buffer = rmm::device_uvector<type_t>(*result.types_buffer, stream);
  auto keys_begin =
    thrust::make_zip_iterator(types_buffer.begin(), geometry_collection_keys.begin());
  auto value_begin = thrust::make_zip_iterator(scatter_map.begin(),
                                               result.lhs_linestring_id->begin(),
                                               result.lhs_segment_id->begin(),
                                               result.rhs_linestring_id->begin(),
                                               result.rhs_segment_id->begin());

  thrust::sort_by_key(rmm::exec_policy(stream), keys_begin, keys_begin + num_geoms, value_begin);

  // Segment-sort the point array
  auto keys_points_begin = thrust::make_zip_iterator(keys_begin, result.points_coords->begin());
  thrust::sort_by_key(rmm::exec_policy(stream),
                      keys_points_begin,
                      keys_points_begin + num_points,
                      scatter_map.begin(),
                      order_key_value_pairs<thrust::tuple<IndexType, IndexType>, vec_2d<T>>{});

  // Segment-sort the segment array
  auto keys_segment_begin =
    thrust::make_zip_iterator(keys_begin + num_points, result.segments_coords->begin());

  thrust::sort_by_key(rmm::exec_policy(stream),
                      keys_segment_begin,
                      keys_segment_begin + num_segments,
                      scatter_map.begin() + num_points,
                      order_key_value_pairs<thrust::tuple<IndexType, IndexType>, segment<T>>{});

  // Restore the order of indices
  auto lhs_linestring_id = std::make_unique<rmm::device_uvector<IndexType>>(num_geoms, stream, mr);
  auto lhs_segment_id    = std::make_unique<rmm::device_uvector<IndexType>>(num_geoms, stream, mr);
  auto rhs_linestring_id = std::make_unique<rmm::device_uvector<IndexType>>(num_geoms, stream, mr);
  auto rhs_segment_id    = std::make_unique<rmm::device_uvector<IndexType>>(num_geoms, stream, mr);

  auto input_it = thrust::make_zip_iterator(result.lhs_linestring_id->begin(),
                                            result.lhs_segment_id->begin(),
                                            result.rhs_linestring_id->begin(),
                                            result.rhs_segment_id->begin());

  auto output_it = thrust::make_zip_iterator(lhs_linestring_id->begin(),
                                             lhs_segment_id->begin(),
                                             rhs_linestring_id->begin(),
                                             rhs_segment_id->begin());

  thrust::scatter(
    rmm::exec_policy(stream), input_it, input_it + num_geoms, scatter_map.begin(), output_it);

  return {std::move(result.geometry_collection_offset),
          std::move(result.types_buffer),
          std::move(result.offset_buffer),
          std::move(result.points_coords),
          std::move(result.segments_coords),
          std::move(lhs_linestring_id),
          std::move(lhs_segment_id),
          std::move(rhs_linestring_id),
          std::move(rhs_segment_id)};
}

template <typename T,
          typename IndexType,
          typename types_t,
          typename point_t   = vec_2d<T>,
          typename segment_t = segment<T>>
auto make_linestring_intersection_result(
  std::initializer_list<IndexType> geometry_collection_offset,
  std::initializer_list<types_t> types_buffer,
  std::initializer_list<IndexType> offset_buffer,
  std::initializer_list<point_t> points_coords,
  std::initializer_list<segment_t> segments_coords,
  std::initializer_list<IndexType> lhs_linestring_ids,
  std::initializer_list<IndexType> lhs_segment_ids,
  std::initializer_list<IndexType> rhs_linestring_ids,
  std::initializer_list<IndexType> rhs_segment_ids,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto d_geometry_collection_offset =
    make_device_uvector<IndexType>(geometry_collection_offset, stream, mr);
  auto d_types_buffer       = make_device_uvector<types_t>(types_buffer, stream, mr);
  auto d_offset_buffer      = make_device_uvector<IndexType>(offset_buffer, stream, mr);
  auto d_points_coords      = make_device_uvector<point_t>(points_coords, stream, mr);
  auto d_segments_coords    = make_device_uvector<segment_t>(segments_coords, stream, mr);
  auto d_lhs_linestring_ids = make_device_uvector<IndexType>(lhs_linestring_ids, stream, mr);
  auto d_lhs_segment_ids    = make_device_uvector<IndexType>(lhs_segment_ids, stream, mr);
  auto d_rhs_linestring_ids = make_device_uvector<IndexType>(rhs_linestring_ids, stream, mr);
  auto d_rhs_segment_ids    = make_device_uvector<IndexType>(rhs_segment_ids, stream, mr);

  return linestring_intersection_result<T, IndexType>{
    std::make_unique<rmm::device_uvector<IndexType>>(d_geometry_collection_offset, stream),
    std::make_unique<rmm::device_uvector<types_t>>(d_types_buffer, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_offset_buffer, stream),
    std::make_unique<rmm::device_uvector<point_t>>(d_points_coords, stream),
    std::make_unique<rmm::device_uvector<segment_t>>(d_segments_coords, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_lhs_linestring_ids, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_lhs_segment_ids, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_rhs_linestring_ids, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_rhs_segment_ids, stream)};
}

}  // namespace test
}  // namespace cuspatial
