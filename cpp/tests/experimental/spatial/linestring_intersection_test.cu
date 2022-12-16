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

#include "intersection_test_utils.cuh"

#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/linestring_intersection.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column.hpp>
#include <cudf_test/column_utilities.hpp>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <initializer_list>
#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

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
  rmm::mr::device_memory_resource* mr,
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
  auto geometry_collection_keys_begin = detail::make_counting_transform_iterator(
    0,
    detail::intersection_functors::offsets_to_keys_functor{
      result.geometry_collection_offset->begin(), result.geometry_collection_offset->end()});
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
  rmm::mr::device_memory_resource* mr)
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

template <typename T>
struct LinestringIntersectionTest : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::mr::device_memory_resource* mr() { return rmm::mr::get_current_device_resource(); }

  template <typename IndexType, typename MultiLinestringRange, typename IntersectionResult>
  void run_single_test(MultiLinestringRange lhs,
                       MultiLinestringRange rhs,
                       IntersectionResult const& expected)
  {
    using types_t = typename IntersectionResult::types_t;

    auto unsorted =
      pairwise_linestring_intersection<T, IndexType>(lhs, rhs, this->mr(), this->stream());
    auto got =
      segment_sort_intersection_result<T, IndexType, types_t>(unsorted, this->mr(), this->stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*std::move(expected.geometry_collection_offset),
                                        *std::move(got.geometry_collection_offset));
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*std::move(expected.types_buffer),
                                        *std::move(got.types_buffer));
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*std::move(expected.offset_buffer),
                                        *std::move(got.offset_buffer));
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*std::move(expected.points_coords),
                                        *std::move(got.points_coords));
    expect_vec_2d_pair_equivalent(*std::move(expected.segments_coords),
                                  *std::move(got.segments_coords));
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*std::move(expected.lhs_linestring_id),
                                        *std::move(got.lhs_linestring_id));
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*std::move(expected.lhs_segment_id),
                                        *std::move(got.lhs_segment_id));
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*std::move(expected.rhs_linestring_id),
                                        *std::move(got.rhs_linestring_id));
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*std::move(expected.rhs_segment_id),
                                        *std::move(got.rhs_segment_id));
  }
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LinestringIntersectionTest, TestTypes);

// // TODO: sort the points in the intersection result since the result order is arbitrary.
TYPED_TEST(LinestringIntersectionTest, Empty)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0}, {0}, std::initializer_list<P>{});

  auto multilinestrings2 = make_multilinestring_array({0}, {0}, std::initializer_list<P>{});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>(
    {0}, {}, {}, {}, {}, {}, {}, {}, {}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, SingletoSingleOnePair)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0.0, 0.0}, P{1.0, 1.0}});

  auto multilinestrings2 = make_multilinestring_array({0, 1}, {0, 2}, {P{0.0, 1.0}, P{1.0, 0.0}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>(
    {0, 1}, {0}, {0}, {P{0.5, 0.5}}, {}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, SingletoSingleOnePairWithDuplicatePoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0.0, 0.0}, P{1.0, 1.0}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1}, {0, 4}, {P{0.0, 1.0}, P{1.0, 0.0}, P{0.5, 0.0}, P{0.5, 1}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>(
    {0, 1}, {0}, {0}, {P{0.5, 0.5}}, {}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, SingletoSingleOnePairWithMergeableSegment)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0.0, 0.0}, P{3.0, 3.0}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1}, {0, 4}, {P{0.0, 0.0}, P{1.0, 1.0}, P{0.5, 0.5}, P{1.5, 1.5}});

  auto expected =
    make_linestring_intersection_result<T, index_t, types_t>({0, 1},
                                                             {1},
                                                             {0},
                                                             {},
                                                             {segment<T>{P{0.0, 0.0}, P{1.5, 1.5}}},
                                                             {0},
                                                             {0},
                                                             {0},
                                                             {0},
                                                             this->stream(),
                                                             this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, SingletoSingleOnePairWithMergeablePoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0.0, 0.0}, P{3.0, 3.0}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1}, {0, 4}, {P{0.0, 1.0}, P{1.0, 0.0}, P{2.0, 2.0}, P{0.0, 0.0}});

  auto expected =
    make_linestring_intersection_result<T, index_t, types_t>({0, 1},
                                                             {1},
                                                             {0},
                                                             {},
                                                             {segment<T>{P{0.0, 0.0}, P{2.0, 2.0}}},
                                                             {0},
                                                             {0},
                                                             {0},
                                                             {2},
                                                             this->stream(),
                                                             this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, TwoPairsSingleToSingle)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 1, 2}, {0, 2, 4}, {P{0.0, 0.0}, P{1.0, 1.0}, P{0.5, 0}, P{0.5, 1}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1, 2}, {0, 2, 4}, {P{0.0, 1.0}, P{1.0, 0.0}, P{0, 0.5}, P{1, 0.5}});

  auto expected =
    make_linestring_intersection_result<T, index_t, types_t>({0, 1, 2},
                                                             {0, 0},
                                                             {0, 1},
                                                             {P{0.5, 0.5}, P{0.5, 0.5}},
                                                             {},
                                                             {0, 0},
                                                             {0, 0},
                                                             {0, 0},
                                                             {0, 0},
                                                             this->stream(),
                                                             this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, TwoPairMultiWithMergeablePoints)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 2, 3},
    {0, 2, 4, 6},
    {P{0.25, 0.25}, P{0.75, 0.75}, P{0.0, 1.0}, P{1.0, 0.0}, P{0.0, 3.0}, P{3.0, 3.0}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1, 2}, {0, 2, 5}, {P{0.0, 0.0}, P{1.0, 1.0}, P{0.0, 2.0}, P{2.0, 4.0}, P{3.0, 2.0}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>(
    {0, 1, 3},
    {1, 0, 0},
    {0, 0, 1},
    {P{1.0, 3.0}, P{2.5, 3.0}},
    {segment<T>{P{0.25, 0.25}, P{0.75, 0.75}}},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 1},
    this->stream(),
    this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, TwoPairMultiWithDuplicatePoints)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 2, 3},
    {0, 2, 4, 6},
    {P{0.0, 1.0}, P{1.0, 0.0}, P{0.5, 0.0}, P{0.5, 1.0}, P{0.0, 3.0}, P{3.0, 3.0}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1, 2}, {0, 2, 4}, {P{0.0, 0.0}, P{1.0, 1.0}, P{0.0, 2.0}, P{2.0, 2.0}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>(
    {0, 1, 1}, {0}, {0}, {P{0.5, 0.5}}, {}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, ManyPairsIntegrated)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1, 2, 3, 4, 5, 6, 7},
                                                      {0, 2, 4, 6, 8, 10, 12, 14},
                                                      {P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 2, 5, 7, 12, 16, 18, 20},
    {P{1, 0},       P{0, 1},     P{0.5, 0},    P{0, 0.5},     P{1, 0.5},
     P{0.5, 0.5},   P{1.5, 1.5}, P{-1, -1},    P{0.25, 0.25}, P{0.25, 0.0},
     P{0.75, 0.75}, P{1.5, 1.5}, P{0.25, 0.0}, P{0.25, 0.5},  P{0.75, 0.75},
     P{1.5, 1.5},   P{2, 2},     P{3, 3},      P{1, 0},       P{2, 0}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>(
    {0, 1, 3, 4, 6, 8, 8, 8},
    {0, 0, 0, 1, 1, 1, 0, 1},
    {0, 1, 2, 0, 1, 2, 3, 3},
    {P{0.5, 0.5}, P{0.25, 0.25}, P{0.5, 0.5}, P{0.25, 0.25}},
    {segment<T>{P{0.5, 0.5}, P{1, 1}},
     segment<T>{P{0, 0}, P{0.25, 0.25}},
     segment<T>{P{0.75, 0.75}, P{1, 1}},
     segment<T>{P{0.75, 0.75}, P{1, 1}}},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 3, 0, 2},
    this->stream(),
    this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, SignedIntegerInput)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, int32_t>::index_t;
  using types_t = typename linestring_intersection_result<T, int32_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1, 2, 3, 4, 5, 6, 7},
                                                      {0, 2, 4, 6, 8, 10, 12, 14},
                                                      {P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 2, 5, 7, 12, 16, 18, 20},
    {P{1, 0},       P{0, 1},     P{0.5, 0},    P{0, 0.5},     P{1, 0.5},
     P{0.5, 0.5},   P{1.5, 1.5}, P{-1, -1},    P{0.25, 0.25}, P{0.25, 0.0},
     P{0.75, 0.75}, P{1.5, 1.5}, P{0.25, 0.0}, P{0.25, 0.5},  P{0.75, 0.75},
     P{1.5, 1.5},   P{2, 2},     P{3, 3},      P{1, 0},       P{2, 0}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>(
    {0, 1, 3, 4, 6, 8, 8, 8},
    {0, 0, 0, 1, 1, 1, 0, 1},
    {0, 1, 2, 0, 1, 2, 3, 3},
    {P{0.5, 0.5}, P{0.25, 0.25}, P{0.5, 0.5}, P{0.25, 0.25}},
    {segment<T>{P{0.5, 0.5}, P{1, 1}},
     segment<T>{P{0, 0}, P{0.25, 0.25}},
     segment<T>{P{0.75, 0.75}, P{1, 1}},
     segment<T>{P{0.75, 0.75}, P{1, 1}}},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 3, 0, 2},
    this->stream(),
    this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}
