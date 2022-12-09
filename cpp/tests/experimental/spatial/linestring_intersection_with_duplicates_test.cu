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

#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/detail/linestring_intersection_with_duplicates.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/iterator/zip_iterator.h>

#include <initializer_list>
#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct LinestringIntersectionDuplicatesTest : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::mr::device_memory_resource* mr() { return rmm::mr::get_current_device_resource(); }

  template <typename IndexType, typename MultilinestringRange1, typename MultilinestringRange2>
  void run_single(MultilinestringRange1 lhs,
                  MultilinestringRange2 rhs,
                  std::initializer_list<IndexType> expected_points_offsets,
                  std::initializer_list<vec_2d<T>> expected_points_coords,
                  std::initializer_list<IndexType> expected_segments_offsets,
                  std::initializer_list<segment<T>> expected_segments_coords,
                  std::initializer_list<IndexType> expected_point_lhs_linestring_ids,
                  std::initializer_list<IndexType> expected_point_lhs_segment_ids,
                  std::initializer_list<IndexType> expected_point_rhs_linestring_ids,
                  std::initializer_list<IndexType> expected_point_rhs_segment_ids,
                  std::initializer_list<IndexType> expected_segment_lhs_linestring_ids,
                  std::initializer_list<IndexType> expected_segment_lhs_segment_ids,
                  std::initializer_list<IndexType> expected_segment_rhs_linestring_ids,
                  std::initializer_list<IndexType> expected_segment_rhs_segment_ids

  )
  {
    auto d_expected_points_offsets   = make_device_vector(expected_points_offsets);
    auto d_expected_points_coords    = make_device_vector(expected_points_coords);
    auto d_expected_segments_offsets = make_device_vector(expected_segments_offsets);
    auto d_expected_segments_coords  = make_device_vector(expected_segments_coords);
    auto d_expected_point_lhs_linestring_ids =
      make_device_vector(expected_point_lhs_linestring_ids);
    auto d_expected_point_lhs_segment_ids = make_device_vector(expected_point_lhs_segment_ids);
    auto d_expected_point_rhs_linestring_ids =
      make_device_vector(expected_point_rhs_linestring_ids);
    auto d_expected_point_rhs_segment_ids = make_device_vector(expected_point_rhs_segment_ids);
    auto d_expected_segment_lhs_linestring_ids =
      make_device_vector(expected_segment_lhs_linestring_ids);
    auto d_expected_segment_lhs_segment_ids = make_device_vector(expected_segment_lhs_segment_ids);
    auto d_expected_segment_rhs_linestring_ids =
      make_device_vector(expected_segment_rhs_linestring_ids);
    auto d_expected_segment_rhs_segment_ids = make_device_vector(expected_segment_rhs_segment_ids);

    auto [points, segments] =
      detail::pairwise_linestring_intersection_with_duplicates<IndexType, T>(
        lhs, rhs, this->mr(), this->stream());

    expect_vector_equivalent(d_expected_points_offsets, *std::move(points.offsets));
    expect_vector_equivalent(d_expected_points_coords, *std::move(points.geoms));
    expect_vector_equivalent(d_expected_segments_offsets, *std::move(segments.offsets));
    expect_segment_equivalent(d_expected_segments_coords, *std::move(segments.geoms));
    expect_vector_equivalent(d_expected_point_lhs_linestring_ids,
                             *std::move(points.lhs_linestring_ids));
    expect_vector_equivalent(d_expected_point_lhs_segment_ids, *std::move(points.lhs_segment_ids));
    expect_vector_equivalent(d_expected_point_rhs_linestring_ids,
                             *std::move(points.rhs_linestring_ids));
    expect_vector_equivalent(d_expected_point_rhs_segment_ids, *std::move(points.rhs_segment_ids));
    expect_vector_equivalent(d_expected_segment_lhs_linestring_ids,
                             *std::move(segments.lhs_linestring_ids));
    expect_vector_equivalent(d_expected_segment_lhs_segment_ids,
                             *std::move(segments.lhs_segment_ids));
    expect_vector_equivalent(d_expected_segment_rhs_linestring_ids,
                             *std::move(segments.rhs_linestring_ids));
    expect_vector_equivalent(d_expected_segment_rhs_segment_ids,
                             *std::move(segments.rhs_segment_ids));
  }
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LinestringIntersectionDuplicatesTest, TestTypes);

// TODO: sort the points in the intersection result since the result order is arbitrary.
TYPED_TEST(LinestringIntersectionDuplicatesTest, Example)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

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

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0, 1, 3, 3, 5, 7, 7, 7},
                     // Expected points
                     {P{0.5, 0.5},
                      P{0.25, 0.25},
                      P{0.5, 0.5},
                      P{0.25, 0.25},
                      P{0.75, 0.75},
                      P{0.25, 0.25},
                      P{0.75, 0.75}},
                     // Segment offsets
                     {0, 0, 0, 1, 3, 4, 4, 4},
                     // Expected segments
                     {segment<T>{P{0.5, 0.5}, P{1, 1}},
                      segment<T>{P{0, 0}, P{0.25, 0.25}},
                      segment<T>{P{0.75, 0.75}, P{1, 1}},
                      segment<T>{P{0.75, 0.75}, P{1, 1}}},
                     // Expected look-back id for points
                     {0, 0, 0, 0, 0, 0, 0},
                     {0, 0, 0, 0, 0, 0, 0},
                     {0, 0, 0, 0, 0, 0, 0},
                     {0, 0, 1, 1, 2, 0, 1},
                     // Expected look-back id for segments
                     {0, 0, 0, 0},
                     {0, 0, 0, 0},
                     {0, 0, 0, 0},
                     {0, 0, 3, 2});
}

// Same Test Case as above, reversing the order of multilinestrings1 and multilinestrings2
TYPED_TEST(LinestringIntersectionDuplicatesTest, ExampleReversed)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

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

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings2.range(),
                     multilinestrings1.range(),
                     // Point Offsets
                     {0, 1, 3, 3, 5, 7, 7, 7},
                     // Expected points
                     {P{0.5, 0.5},
                      P{0.25, 0.25},
                      P{0.5, 0.5},
                      P{0.25, 0.25},
                      P{0.75, 0.75},
                      P{0.25, 0.25},
                      P{0.75, 0.75}},
                     // Segment Offsets
                     {0, 0, 0, 1, 3, 4, 4, 4},
                     // Expected Segments
                     {segment<T>{P{0.5, 0.5}, P{1, 1}},
                      segment<T>{P{0, 0}, P{0.25, 0.25}},
                      segment<T>{P{0.75, 0.75}, P{1, 1}},
                      segment<T>{P{0.75, 0.75}, P{1, 1}}},
                     // Point look-back ids
                     {0, 0, 0, 0, 0, 0, 0},
                     {0, 0, 1, 1, 2, 0, 1},
                     {0, 0, 0, 0, 0, 0, 0},
                     {0, 0, 0, 0, 0, 0, 0},
                     // Segment look-back ids
                     {0, 0, 0, 0},
                     {0, 0, 3, 2},
                     {0, 0, 0, 0},
                     {0, 0, 0, 0});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, MultilinestringsIntersectionWithDuplicates)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 2, 5},
    {0, 2, 4, 6, 8, 10},
    {P{0, 0}, P{1, 1}, P{1, 0}, P{2, 1}, P{0, 2}, P{1, 2}, P{0, 3}, P{0, 2}, P{0, 3}, P{1, 2}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1, 2}, {0, 2, 4}, {P{0, 1}, P{2, 0}, P{0, 2.5}, P{1, 2.5}});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Points offsets
                     {0, 2, 4},
                     // Expected Points
                     {P{2 / 3., 2 / 3.}, P{4 / 3., 1 / 3.}, P{0, 2.5}, P{0.5, 2.5}},
                     // Segment offsets
                     {0, 0, 0},
                     // Expected Segments
                     {},
                     // Point look-back ids
                     {0, 1, 1, 2},
                     {0, 0, 0, 0},
                     {0, 0, 0, 0},
                     {0, 0, 0, 0},
                     // Segment look-back ids
                     {},
                     {},
                     {},
                     {});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, Empty)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array({0}, {0}, std::initializer_list<P>{});

  auto multilinestrings2 = make_multilinestring_array({0}, {0}, std::initializer_list<P>{});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0},
                     // Expected Points
                     {},
                     // Segment offsets
                     {0},
                     // Expected segments
                     {},
                     // Point look-back ids
                     {},
                     {},
                     {},
                     {},
                     // segment look-back ids
                     {},
                     {},
                     {},
                     {});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, OnePairSingleToSingleOneSegment)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 1}, P{1, 0}});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0, 1},
                     // Expected Points
                     {P{0.5, 0.5}},
                     // Segment offsets
                     {0, 0},
                     // Expected segments
                     {},
                     // Point look-back ids
                     {0},
                     {0},
                     {0},
                     {0},
                     // segment look-back ids
                     {},
                     {},
                     {},
                     {});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, OnePairSingleToSingleTwoSegments)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 3}, {P{-1, 0}, P{0, 0}, P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array({0, 1}, {0, 3}, {P{0, 2}, P{0, 1}, P{1, 0}});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0, 1},
                     // Expected Points
                     {P{0.5, 0.5}},
                     // Segment offsets
                     {0, 0},
                     // Expected segments
                     {},
                     // Point look-back ids
                     {0},
                     {1},
                     {0},
                     {1},
                     // segment look-back ids
                     {},
                     {},
                     {},
                     {});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, OnePairSingletoSingleOverlap)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1}, {0, 2}, {P{0.75, 0.75}, P{0.25, 0.25}});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0, 0},
                     // Expected Points
                     {},
                     // Segment offsets
                     {0, 1},
                     // Expected segments
                     {segment<T>{P{0.25, 0.25}, P{0.75, 0.75}}},
                     // Point look-back ids
                     {},
                     {},
                     {},
                     {},
                     // segment look-back ids
                     {0},
                     {0},
                     {0},
                     {0});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, OnePairSingletoSingleOverlapTwoSegments)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 3}, {P{0, 0}, P{1, 1}, P{2, 2}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1}, {0, 3}, {P{1.25, 1.25}, P{0.75, 0.75}, P{0.25, 0.25}});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0, 0},
                     // Expected Points
                     {},
                     // Segment offsets
                     {0, 3},
                     // Expected segments
                     {
                       segment<T>{P{0.75, 0.75}, P{1.0, 1.0}},
                       segment<T>{P{1.0, 1.0}, P{1.25, 1.25}},
                       segment<T>{P{0.25, 0.25}, P{0.75, 0.75}},
                     },
                     // Point look-back ids
                     {},
                     {},
                     {},
                     {},
                     // segment look-back ids
                     {0, 0, 0},
                     {0, 1, 0},
                     {0, 0, 0},
                     {0, 0, 1});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, OnePairMultiSingle)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 =
    make_multilinestring_array({0, 2}, {0, 2, 4}, {P{0, 0}, P{1, 1}, P{0, 0}, P{-1, 1}});

  auto multilinestrings2 = make_multilinestring_array({0, 1}, {0, 2}, {P{-2, 0.5}, P{2, 0.5}});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0, 2},
                     // Expected Points
                     {P{0.5, 0.5}, P{-0.5, 0.5}},
                     // Segment offsets
                     {0, 0},
                     // Expected segments
                     {},
                     // Point look-back ids
                     {0, 1},
                     {0, 0},
                     {0, 0},
                     {0, 0},
                     // segment look-back ids
                     {},
                     {},
                     {},
                     {});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, TwoPairsSingletoSingle)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 =
    make_multilinestring_array({0, 1, 2}, {0, 2, 4}, {P{0, 0}, P{1, 1}, P{0, 0}, P{-1, 1}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1, 2}, {0, 2, 4}, {P{0, 1}, P{1, 0}, P{0, 1}, P{-1, 0}});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0, 1, 2},
                     // Expected Points
                     {P{0.5, 0.5}, P{-0.5, 0.5}},
                     // Segment offsets
                     {0, 0, 0},
                     // Expected segments
                     {},
                     // Point look-back ids
                     {0, 0},
                     {0, 0},
                     {0, 0},
                     {0, 0},
                     // segment look-back ids
                     {},
                     {},
                     {},
                     {});
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, TwoPairsMultitoMulti)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 2, 4},
    {0, 2, 4, 6, 8},
    {P{0, 0}, P{1, 1}, P{2, 0}, P{3, 1}, P{1, 0}, P{1, 1}, P{0, 0}, P{1, 0}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 2, 4},
    {0, 2, 4, 6, 8},
    {P{0, 1}, P{1, 0}, P{2, 1}, P{3, 0}, P{-1, 0}, P{-2, 1}, P{-3, -2}, P{-4, 3}});

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     multilinestrings1.range(),
                     multilinestrings2.range(),
                     // Point offsets
                     {0, 2, 2},
                     // Expected Points
                     {P{0.5, 0.5}, P{2.5, 0.5}},
                     // Segment offsets
                     {0, 0, 0},
                     // Expected segments
                     {},
                     // Point look-back ids
                     {0, 1},
                     {0, 0},
                     {0, 1},
                     {0, 0},
                     // segment look-back ids
                     {},
                     {},
                     {},
                     {});
}
