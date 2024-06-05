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

#include "intersection_test_utils.cuh"

#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/intersection.cuh>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/traits.hpp>

#include <cudf/column/column.hpp>
#include <cudf_test/column_utilities.hpp>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <initializer_list>
#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct LinestringIntersectionTest : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::device_async_resource_ref mr() { return rmm::mr::get_current_device_resource(); }

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

// float and double are logically the same but would require separate tests due to precision.
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

TYPED_TEST(LinestringIntersectionTest, OnePairWithRings)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array<T>({0, 1}, {0, 2}, {{-1, 0}, {0, 0}});

  auto multilinestrings2 =
    make_multilinestring_array<T>({0, 1}, {0, 5}, {{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>(
    {0, 1}, {0}, {0}, {P{0.0, 0.0}}, {}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

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

TYPED_TEST(LinestringIntersectionTest, ThreePairIdenticalInputsNoRing)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 1, 2, 3},
    {0, 2, 4, 6},
    {P{0.0, 0.0}, P{1.0, 1.0}, P{0.0, 0.0}, P{1.0, 1.0}, P{0.0, 0.0}, P{1.0, 1.0}});

  auto multilinestrings2 = make_multilinestring_array<T>({0, 1, 2, 3},
                                                         {0, 4, 8, 12},
                                                         {{0, 0},
                                                          {0, 1},
                                                          {1, 1},
                                                          {1, 0},

                                                          {0, 0},
                                                          {0, 1},
                                                          {1, 1},
                                                          {1, 0},

                                                          {0, 0},
                                                          {0, 1},
                                                          {1, 1},
                                                          {1, 0}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>({0, 2, 4, 6},
                                                                           {0, 0, 0, 0, 0, 0},
                                                                           {0, 1, 2, 3, 4, 5},
                                                                           {
                                                                             {0.0, 0.0},
                                                                             {1.0, 1.0},
                                                                             {0.0, 0.0},
                                                                             {1.0, 1.0},
                                                                             {0.0, 0.0},
                                                                             {1.0, 1.0},
                                                                           },
                                                                           {},
                                                                           {0, 0, 0, 0, 0, 0},
                                                                           {0, 0, 0, 0, 0, 0},
                                                                           {0, 0, 0, 0, 0, 0},
                                                                           {0, 1, 0, 1, 0, 1},
                                                                           this->stream(),
                                                                           this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single_test<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, ThreePairIdenticalInputsHasRing)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = typename linestring_intersection_result<T, std::size_t>::index_t;
  using types_t = typename linestring_intersection_result<T, std::size_t>::types_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 1, 2, 3},
    {0, 2, 4, 6},
    {P{0.0, 0.0}, P{1.0, 1.0}, P{0.0, 0.0}, P{1.0, 1.0}, P{0.0, 0.0}, P{1.0, 1.0}});

  auto multilinestrings2 = make_multilinestring_array<T>({0, 1, 2, 3},
                                                         {0, 5, 10, 15},
                                                         {{0, 0},
                                                          {0, 1},
                                                          {1, 1},
                                                          {1, 0},
                                                          {0, 0},
                                                          {0, 0},
                                                          {0, 1},
                                                          {1, 1},
                                                          {1, 0},
                                                          {0, 0},
                                                          {0, 0},
                                                          {0, 1},
                                                          {1, 1},
                                                          {1, 0},
                                                          {0, 0}});

  auto expected = make_linestring_intersection_result<T, index_t, types_t>({0, 2, 4, 6},
                                                                           {0, 0, 0, 0, 0, 0},
                                                                           {0, 1, 2, 3, 4, 5},
                                                                           {
                                                                             {0.0, 0.0},
                                                                             {1.0, 1.0},
                                                                             {0.0, 0.0},
                                                                             {1.0, 1.0},
                                                                             {0.0, 0.0},
                                                                             {1.0, 1.0},
                                                                           },
                                                                           {},
                                                                           {0, 0, 0, 0, 0, 0},
                                                                           {0, 0, 0, 0, 0, 0},
                                                                           {0, 0, 0, 0, 0, 0},
                                                                           {0, 1, 0, 1, 0, 1},
                                                                           this->stream(),
                                                                           this->mr());

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
