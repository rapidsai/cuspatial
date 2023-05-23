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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/geometry/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct MultipolygonRangeTest : public BaseFixture {
  void run_multipolygon_segment_method_iterator_single(
    std::initializer_list<std::size_t> geometry_offset,
    std::initializer_list<std::size_t> part_offset,
    std::initializer_list<std::size_t> ring_offset,
    std::initializer_list<vec_2d<T>> coordinates,
    std::initializer_list<segment<T>> expected)
  {
    auto multipolygon_array =
      make_multipolygon_array(geometry_offset, part_offset, ring_offset, coordinates);
    auto rng           = multipolygon_array.range();
    auto segments      = rng._segments(stream());
    auto segment_range = segments.segment_range();

    auto got = rmm::device_uvector<segment<T>>(segment_range.num_segments(), stream());

    thrust::copy(
      rmm::exec_policy(stream()), segment_range.begin(), segment_range.end(), got.begin());

    auto d_expected = thrust::device_vector<segment<T>>(expected.begin(), expected.end());

    CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(got, d_expected);
  }

  void run_multipolygon_point_count_iterator_single(
    std::initializer_list<std::size_t> geometry_offset,
    std::initializer_list<std::size_t> part_offset,
    std::initializer_list<std::size_t> ring_offset,
    std::initializer_list<vec_2d<T>> coordinates,
    std::initializer_list<std::size_t> expected_point_counts)
  {
    auto multipolygon_array =
      make_multipolygon_array(geometry_offset, part_offset, ring_offset, coordinates);
    auto rng = multipolygon_array.range();

    auto got = rmm::device_uvector<std::size_t>(rng.num_multipolygons(), stream());

    thrust::copy(rmm::exec_policy(stream()),
                 rng.multipolygon_point_count_begin(),
                 rng.multipolygon_point_count_end(),
                 got.begin());

    auto d_expected = thrust::device_vector<std::size_t>(expected_point_counts.begin(),
                                                         expected_point_counts.end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(got, d_expected);
  }

  void run_multipolygon_segment_method_count_single(
    std::initializer_list<std::size_t> geometry_offset,
    std::initializer_list<std::size_t> part_offset,
    std::initializer_list<std::size_t> ring_offset,
    std::initializer_list<vec_2d<T>> coordinates,
    std::initializer_list<std::size_t> expected_segment_counts)
  {
    auto multipolygon_array =
      make_multipolygon_array(geometry_offset, part_offset, ring_offset, coordinates);
    auto rng           = multipolygon_array.range();
    auto segments      = rng._segments(stream());
    auto segment_range = segments.segment_range();

    auto got = rmm::device_uvector<std::size_t>(rng.num_multipolygons(), stream());

    thrust::copy(rmm::exec_policy(stream()),
                 segment_range.multigeometry_count_begin(),
                 segment_range.multigeometry_count_end(),
                 got.begin());

    auto d_expected = thrust::device_vector<std::size_t>(expected_segment_counts.begin(),
                                                         expected_segment_counts.end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(got, d_expected);
  }

  void test_multipolygon_as_multilinestring(
    std::initializer_list<std::size_t> multipolygon_geometry_offset,
    std::initializer_list<std::size_t> multipolygon_part_offset,
    std::initializer_list<std::size_t> ring_offset,
    std::initializer_list<vec_2d<T>> multipolygon_coordinates,
    std::initializer_list<std::size_t> multilinestring_geometry_offset,
    std::initializer_list<std::size_t> multilinestring_part_offset,
    std::initializer_list<vec_2d<T>> multilinestring_coordinates)
  {
    auto multipolygon_array = make_multipolygon_array(multipolygon_geometry_offset,
                                                      multipolygon_part_offset,
                                                      ring_offset,
                                                      multipolygon_coordinates);
    auto rng                = multipolygon_array.range().as_multilinestring_range();

    auto got =
      make_multilinestring_array(range(rng.geometry_offsets_begin(), rng.geometry_offsets_end()),
                                 range(rng.part_offsets_begin(), rng.part_offsets_end()),
                                 range(rng.point_begin(), rng.point_end()));

    auto expected = make_multilinestring_array(
      multilinestring_geometry_offset, multilinestring_part_offset, multilinestring_coordinates);

    CUSPATIAL_EXPECT_MULTILINESTRING_ARRAY_EQUIVALENT(expected, got);
  }

  void test_multipolygon_as_multipoint(
    std::initializer_list<std::size_t> multipolygon_geometry_offset,
    std::initializer_list<std::size_t> multipolygon_part_offset,
    std::initializer_list<std::size_t> ring_offset,
    std::initializer_list<vec_2d<T>> multipolygon_coordinates,
    std::initializer_list<std::size_t> multipoint_geometry_offset,
    std::initializer_list<vec_2d<T>> multipoint_coordinates)
  {
    auto multipolygon_array = make_multipolygon_array(multipolygon_geometry_offset,
                                                      multipolygon_part_offset,
                                                      ring_offset,
                                                      multipolygon_coordinates);
    auto rng                = multipolygon_array.range().as_multipoint_range();

    auto got = make_multipoints_array(range(rng.offsets_begin(), rng.offsets_end()),
                                      range(rng.point_begin(), rng.point_end()));

    auto expected = make_multipoints_array(
      range(multipoint_geometry_offset.begin(), multipoint_geometry_offset.end()),
      range(multipoint_coordinates.begin(), multipoint_coordinates.end()));

    CUSPATIAL_EXPECT_MULTIPOINT_ARRAY_EQUIVALENT(expected, got);
  }
};

TYPED_TEST_CASE(MultipolygonRangeTest, FloatingPointTypes);

TYPED_TEST(MultipolygonRangeTest, SegmentIterators)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_iterator_single,
                     {0, 1},
                     {0, 1},
                     {0, 4},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}},
                     {S{{0, 0}, P{1, 0}}, S{P{1, 0}, P{1, 1}}, S{P{1, 1}, P{0, 0}}});
}

TYPED_TEST(MultipolygonRangeTest, SegmentIterators2)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_iterator_single,
                     {0, 1},
                     {0, 2},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}},
                     {{{0, 0}, {1, 0}},
                      {{1, 0}, {1, 1}},
                      {{1, 1}, {0, 0}},
                      {{10, 10}, {11, 10}},
                      {{11, 10}, {11, 11}},
                      {{11, 11}, {10, 10}}});
}

TYPED_TEST(MultipolygonRangeTest, SegmentIterators3)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_iterator_single,
                     {0, 2},
                     {0, 1, 2},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}},
                     {{{0, 0}, {1, 0}},
                      {{1, 0}, {1, 1}},
                      {{1, 1}, {0, 0}},
                      {{10, 10}, {11, 10}},
                      {{11, 10}, {11, 11}},
                      {{11, 11}, {10, 10}}});
}

TYPED_TEST(MultipolygonRangeTest, SegmentIterators4)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_iterator_single,
                     {0, 1, 2},
                     {0, 1, 2},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}},
                     {{{0, 0}, {1, 0}},
                      {{1, 0}, {1, 1}},
                      {{1, 1}, {0, 0}},
                      {{10, 10}, {11, 10}},
                      {{11, 10}, {11, 11}},
                      {{11, 11}, {10, 10}}});
}

TYPED_TEST(MultipolygonRangeTest, SegmentIterators5)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_iterator_single,
                     {0, 1, 2, 3},
                     {0, 1, 2, 3},
                     {0, 4, 9, 14},
                     {{-1, -1},
                      {-2, -2},
                      {-2, -1},
                      {-1, -1},

                      {-20, -20},
                      {-20, -21},
                      {-21, -21},
                      {-21, -20},
                      {-20, -20},

                      {-10, -10},
                      {-10, -11},
                      {-11, -11},
                      {-11, -10},
                      {-10, -10}},

                     {{{-1, -1}, {-2, -2}},
                      {{-2, -2}, {-2, -1}},
                      {{-2, -1}, {-1, -1}},
                      {{-20, -20}, {-20, -21}},
                      {{-20, -21}, {-21, -21}},
                      {{-21, -21}, {-21, -20}},
                      {{-21, -20}, {-20, -20}},
                      {{-10, -10}, {-10, -11}},
                      {{-10, -11}, {-11, -11}},
                      {{-11, -11}, {-11, -10}},
                      {{-11, -10}, {-10, -10}}});
}

TYPED_TEST(MultipolygonRangeTest, SegmentIterators5EmptyRing)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_iterator_single,
                     {0, 1, 2},
                     {0, 1, 3},
                     {0, 4, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}},
                     {{{0, 0}, {1, 0}},
                      {{1, 0}, {1, 1}},
                      {{1, 1}, {0, 0}},
                      {{10, 10}, {11, 10}},
                      {{11, 10}, {11, 11}},
                      {{11, 11}, {10, 10}}});
}

TYPED_TEST(MultipolygonRangeTest, SegmentIterators6EmptyPolygon)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_iterator_single,
                     {0, 1, 3},
                     {0, 1, 1, 2},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}},
                     {{{0, 0}, {1, 0}},
                      {{1, 0}, {1, 1}},
                      {{1, 1}, {0, 0}},
                      {{10, 10}, {11, 10}},
                      {{11, 10}, {11, 11}},
                      {{11, 11}, {10, 10}}});
}

TYPED_TEST(MultipolygonRangeTest, SegmentIterators7EmptyMultiPolygon)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_iterator_single,
                     {0, 1, 1, 2},
                     {0, 1, 2},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}},
                     {{{0, 0}, {1, 0}},
                      {{1, 0}, {1, 1}},
                      {{1, 1}, {0, 0}},
                      {{10, 10}, {11, 10}},
                      {{11, 10}, {11, 11}},
                      {{11, 11}, {10, 10}}});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonCountIterator)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_point_count_iterator_single,
                     {0, 1},
                     {0, 1},
                     {0, 4},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}},
                     {4});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonCountIterator2)
{
  CUSPATIAL_RUN_TEST(
    this->run_multipolygon_point_count_iterator_single,
    {0, 1},
    {0, 2},
    {0, 4, 8},
    {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {0.2, 0.2}, {0.2, 0.3}, {0.3, 0.3}, {0.3, 0.2}},
    {8});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonCountIterator3)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_point_count_iterator_single,
                     {0, 2},
                     {0, 2, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {0.2, 0.2},
                      {0.2, 0.3},
                      {0.3, 0.3},
                      {0.3, 0.2},
                      {0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 1}},
                     {12});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonCountIterator4)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_point_count_iterator_single,
                     {0, 2, 3},
                     {0, 2, 3, 4},
                     {0, 4, 8, 12, 16},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {0.2, 0.2},
                      {0.2, 0.3},
                      {0.3, 0.3},
                      {0.2, 0.2},
                      {0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0}},
                     {12, 4});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonSegmentCount)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_count_single,
                     {0, 1},
                     {0, 1},
                     {0, 4},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}},
                     {3});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonSegmentCount2)
{
  CUSPATIAL_RUN_TEST(
    this->run_multipolygon_segment_method_count_single,
    {0, 1},
    {0, 2},
    {0, 4, 8},
    {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {0.2, 0.2}, {0.2, 0.3}, {0.3, 0.3}, {0.2, 0.2}},
    {6});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonSegmentCount3)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_count_single,
                     {0, 2},
                     {0, 2, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {0.2, 0.2},
                      {0.2, 0.3},
                      {0.3, 0.3},
                      {0.2, 0.2},
                      {0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0}},
                     {9});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonSegmentCount4)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_count_single,
                     {0, 2, 3},
                     {0, 2, 3, 4},
                     {0, 4, 8, 12, 16},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {0.2, 0.2},
                      {0.2, 0.3},
                      {0.3, 0.3},
                      {0.2, 0.2},
                      {0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0}},
                     {9, 3});
}

// FIXME: multipolygon constructor doesn't allow empty rings, should it?
TYPED_TEST(MultipolygonRangeTest, MultipolygonSegmentCount_ContainsEmptyRing)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_count_single,
                     {0, 2, 3},
                     {0, 2, 3, 4},
                     {0, 7, 7, 11, 18},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0.5, 1.5},
                      {0, 1.0},
                      {0.5, 0.5},
                      {0, 0},
                      {0.2, 0.2},
                      {0.2, 0.3},
                      {0.3, 0.3},
                      {0.2, 0.2},
                      {0, 0},
                      {1, 0},
                      {1, 1},
                      {0.5, 1.5},
                      {0, 1.0},
                      {0.5, 0.5},
                      {0, 0}},
                     {9, 6});
}

// FIXME: multipolygon constructor doesn't allow empty rings, should it?
TYPED_TEST(MultipolygonRangeTest, MultipolygonSegmentCount_ContainsEmptyPart)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_method_count_single,
                     {0, 3, 4},
                     {0, 1, 1, 2, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {0.2, 0.2},
                      {0.2, 0.3},
                      {0.3, 0.3},
                      {0.2, 0.2},
                      {0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0}},
                     {6, 3});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultilinestring1)
{
  CUSPATIAL_RUN_TEST(this->test_multipolygon_as_multilinestring,
                     {0, 1, 2},
                     {0, 1, 2},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}},
                     {0, 1, 2},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultilinestring2)
{
  CUSPATIAL_RUN_TEST(this->test_multipolygon_as_multilinestring,
                     {0, 1, 2},
                     {0, 1, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {10, 10},
                      {11, 10},
                      {11, 11},
                      {10, 10},
                      {20, 20},
                      {21, 20},
                      {21, 21},
                      {20, 20}},
                     {0, 1, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {10, 10},
                      {11, 10},
                      {11, 11},
                      {10, 10},
                      {20, 20},
                      {21, 20},
                      {21, 21},
                      {20, 20}});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultilinestring3)
{
  CUSPATIAL_RUN_TEST(this->test_multipolygon_as_multilinestring,
                     {0, 1, 2},
                     {0, 2, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {10, 10},
                      {11, 10},
                      {11, 11},
                      {10, 10},
                      {20, 20},
                      {21, 20},
                      {21, 21},
                      {20, 20}},
                     {0, 2, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {10, 10},
                      {11, 10},
                      {11, 11},
                      {10, 10},
                      {20, 20},
                      {21, 20},
                      {21, 21},
                      {20, 20}});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultiPoint1)
{
  CUSPATIAL_RUN_TEST(this->test_multipolygon_as_multipoint,
                     {0, 1, 2},
                     {0, 1, 2},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}},
                     {0, 4, 8},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultiPoint2)
{
  CUSPATIAL_RUN_TEST(this->test_multipolygon_as_multipoint,
                     {0, 1, 2},
                     {0, 1, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {10, 10},
                      {11, 10},
                      {11, 11},
                      {10, 10},
                      {20, 20},
                      {21, 20},
                      {21, 21},
                      {20, 20}},
                     {0, 4, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {10, 10},
                      {11, 10},
                      {11, 11},
                      {10, 10},
                      {20, 20},
                      {21, 20},
                      {21, 21},
                      {20, 20}});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultiPoint3)
{
  CUSPATIAL_RUN_TEST(this->test_multipolygon_as_multipoint,
                     {0, 1, 2},
                     {0, 2, 3},
                     {0, 4, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {10, 10},
                      {11, 10},
                      {11, 11},
                      {10, 10},
                      {20, 20},
                      {21, 20},
                      {21, 21},
                      {20, 20}},
                     {0, 8, 12},
                     {{0, 0},
                      {1, 0},
                      {1, 1},
                      {0, 0},
                      {10, 10},
                      {11, 10},
                      {11, 11},
                      {10, 10},
                      {20, 20},
                      {21, 20},
                      {21, 21},
                      {20, 20}});
}
