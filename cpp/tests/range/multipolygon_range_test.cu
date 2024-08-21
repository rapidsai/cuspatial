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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/geometry/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/sequence.h>
#include <thrust/tabulate.h>

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

    auto geometry_offsets =
      rmm::device_vector<std::size_t>(rng.geometry_offset_begin(), rng.geometry_offset_end());
    auto part_offsets =
      rmm::device_vector<std::size_t>(rng.part_offset_begin(), rng.part_offset_end());
    auto points = rmm::device_vector<vec_2d<T>>(rng.point_begin(), rng.point_end());

    auto got = make_multilinestring_array(
      std::move(geometry_offsets), std::move(part_offsets), std::move(points));

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

    auto got = make_multipoint_array(range(rng.offsets_begin(), rng.offsets_end()),
                                     range(rng.point_begin(), rng.point_end()));

    auto expected = make_multipoint_array(
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

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultipolygon1)
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

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultipolygon2)
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

TYPED_TEST(MultipolygonRangeTest, MultipolygonAsMultipolygon3)
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

template <typename MultiPolygonRange, typename PointOutputIt>
CUSPATIAL_KERNEL void array_access_tester(MultiPolygonRange rng,
                                          std::size_t i,
                                          PointOutputIt output)
{
  thrust::copy(thrust::seq, rng[i].point_begin(), rng[i].point_end(), output);
}

template <typename T>
class MultipolygonRangeTestBase : public BaseFixture {
 public:
  struct copy_leading_point_functor {
    template <typename MultiPolygonRef>
    __device__ vec_2d<T> operator()(MultiPolygonRef mpolygon)
    {
      return mpolygon.size() > 0 ? mpolygon.point_begin()[0] : vec_2d<T>{-1, -1};
    }
  };

  template <typename MultiPolygonRange>
  struct ring_idx_from_point_idx_functor {
    MultiPolygonRange mpolygons;
    __device__ std::size_t operator()(std::size_t point_idx)
    {
      return mpolygons.ring_idx_from_point_idx(point_idx);
    }
  };

  template <typename MultiPolygonRange>
  struct part_idx_from_ring_idx_functor {
    MultiPolygonRange mpolygons;
    __device__ std::size_t operator()(std::size_t ring_idx)
    {
      return mpolygons.part_idx_from_ring_idx(ring_idx);
    }
  };

  template <typename MultiPolygonRange>
  struct geometry_idx_from_part_idx_functor {
    MultiPolygonRange mpolygons;
    __device__ std::size_t operator()(std::size_t part_idx)
    {
      return mpolygons.geometry_idx_from_part_idx(part_idx);
    }
  };

  void SetUp() { make_test_multipolygon(); }

  virtual void make_test_multipolygon() = 0;

  auto range() { return test_multipolygon->range(); }

  void run_test()
  {
    test_size();

    test_num_multipolygons();

    test_num_polygons();

    test_num_rings();

    test_num_points();

    test_multipolygon_it();

    test_begin();

    test_end();

    test_point_it();

    test_geometry_offsets_it();

    test_part_offset_it();

    test_ring_offset_it();

    test_ring_idx_from_point_idx();

    test_part_idx_from_ring_idx();

    test_geometry_idx_from_part_idx();

    test_array_access_operator();

    test_multipolygon_point_count_it();

    test_multipolygon_ring_count_it();
  }

  void test_size() { EXPECT_EQ(range().size(), range().num_multipolygons()); };

  virtual void test_num_multipolygons() = 0;

  virtual void test_num_polygons() = 0;

  virtual void test_num_rings() = 0;

  virtual void test_num_points() = 0;

  virtual void test_multipolygon_it() = 0;

  void test_begin() { EXPECT_EQ(range().begin(), range().multipolygon_begin()); }

  void test_end() { EXPECT_EQ(range().end(), range().multipolygon_end()); }

  virtual void test_point_it() = 0;

  virtual void test_geometry_offsets_it() = 0;

  virtual void test_part_offset_it() = 0;

  virtual void test_ring_offset_it() = 0;

  virtual void test_ring_idx_from_point_idx() = 0;

  virtual void test_part_idx_from_ring_idx() = 0;

  virtual void test_geometry_idx_from_part_idx() = 0;

  virtual void test_array_access_operator() = 0;

  virtual void test_multipolygon_point_count_it() = 0;

  virtual void test_multipolygon_ring_count_it() = 0;

  // helper method to access multipolygon range
  rmm::device_uvector<vec_2d<T>> copy_leading_point_multipolygon()
  {
    auto rng      = range();
    auto d_points = rmm::device_uvector<vec_2d<T>>(rng.num_multipolygons(), stream());
    thrust::transform(rmm::exec_policy(stream()),
                      rng.multipolygon_begin(),
                      rng.multipolygon_end(),
                      d_points.begin(),
                      copy_leading_point_functor{});
    return d_points;
  }

  rmm::device_uvector<vec_2d<T>> copy_all_points()
  {
    auto rng      = range();
    auto d_points = rmm::device_uvector<vec_2d<T>>(rng.num_points(), stream());
    thrust::copy(rmm::exec_policy(stream()), rng.point_begin(), rng.point_end(), d_points.begin());
    return d_points;
  }

  rmm::device_uvector<std::size_t> copy_geometry_offsets()
  {
    auto rng       = range();
    auto d_offsets = rmm::device_uvector<std::size_t>(rng.num_multipolygons() + 1, stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.geometry_offset_begin(),
                 rng.geometry_offset_end(),
                 d_offsets.begin());
    return d_offsets;
  }

  rmm::device_uvector<std::size_t> copy_part_offsets()
  {
    auto rng       = range();
    auto d_offsets = rmm::device_uvector<std::size_t>(rng.num_polygons() + 1, stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.part_offset_begin(),
                 rng.part_offset_end(),
                 d_offsets.begin());
    return d_offsets;
  }

  rmm::device_uvector<std::size_t> copy_ring_offsets()
  {
    auto rng       = range();
    auto d_offsets = rmm::device_uvector<std::size_t>(rng.num_rings() + 1, stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.ring_offset_begin(),
                 rng.ring_offset_end(),
                 d_offsets.begin());
    return d_offsets;
  }

  rmm::device_uvector<std::size_t> copy_ring_idx_from_point_idx()
  {
    auto rng        = range();
    auto d_ring_idx = rmm::device_uvector<std::size_t>(rng.num_points(), stream());
    thrust::tabulate(rmm::exec_policy(stream()),
                     d_ring_idx.begin(),
                     d_ring_idx.end(),
                     ring_idx_from_point_idx_functor<decltype(rng)>{rng});
    return d_ring_idx;
  }

  rmm::device_uvector<std::size_t> copy_part_idx_from_ring_idx()
  {
    auto rng        = range();
    auto d_part_idx = rmm::device_uvector<std::size_t>(rng.num_rings(), stream());
    thrust::tabulate(rmm::exec_policy(stream()),
                     d_part_idx.begin(),
                     d_part_idx.end(),
                     part_idx_from_ring_idx_functor<decltype(rng)>{rng});
    return d_part_idx;
  }

  rmm::device_uvector<std::size_t> copy_geometry_idx_from_part_idx()
  {
    auto rng            = range();
    auto d_geometry_idx = rmm::device_uvector<std::size_t>(rng.num_polygons(), stream());
    thrust::tabulate(rmm::exec_policy(stream()),
                     d_geometry_idx.begin(),
                     d_geometry_idx.end(),
                     geometry_idx_from_part_idx_functor<decltype(rng)>{rng});
    return d_geometry_idx;
  }

  rmm::device_uvector<vec_2d<T>> copy_all_points_of_ith_multipolygon(std::size_t i)
  {
    auto rng = this->range();
    rmm::device_scalar<std::size_t> num_points(stream());

    thrust::copy_n(
      rmm::exec_policy(stream()), rng.multipolygon_point_count_begin() + i, 1, num_points.data());

    auto d_all_points = rmm::device_uvector<vec_2d<T>>(num_points.value(stream()), stream());

    array_access_tester<<<1, 1, 0, stream()>>>(rng, i, d_all_points.data());
    return d_all_points;
  }

  rmm::device_uvector<std::size_t> copy_multipolygon_point_count()
  {
    auto rng           = this->range();
    auto d_point_count = rmm::device_uvector<std::size_t>(rng.num_multipolygons(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multipolygon_point_count_begin(),
                 rng.multipolygon_point_count_end(),
                 d_point_count.begin());
    return d_point_count;
  }

  rmm::device_uvector<std::size_t> copy_multipolygon_ring_count()
  {
    auto rng          = this->range();
    auto d_ring_count = rmm::device_uvector<std::size_t>(rng.num_multipolygons(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multipolygon_ring_count_begin(),
                 rng.multipolygon_ring_count_end(),
                 d_ring_count.begin());
    return d_ring_count;
  }

 protected:
  std::unique_ptr<multipolygon_array<rmm::device_vector<std::size_t>,
                                     rmm::device_vector<std::size_t>,
                                     rmm::device_vector<std::size_t>,
                                     rmm::device_vector<vec_2d<T>>>>
    test_multipolygon;
};

template <typename T>
class MultipolygonRangeEmptyTest : public MultipolygonRangeTestBase<T> {
  void make_test_multipolygon()
  {
    auto geometry_offsets = make_device_vector<std::size_t>({0});
    auto part_offsets     = make_device_vector<std::size_t>({0});
    auto ring_offsets     = make_device_vector<std::size_t>({0});
    auto coordinates      = make_device_vector<vec_2d<T>>({});

    this->test_multipolygon = std::make_unique<multipolygon_array<rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<vec_2d<T>>>>(
      std::move(geometry_offsets),
      std::move(part_offsets),
      std::move(ring_offsets),
      std::move(coordinates));
  }

  void test_num_multipolygons() { EXPECT_EQ(this->range().num_multipolygons(), 0); }

  void test_num_polygons() { EXPECT_EQ(this->range().num_polygons(), 0); }

  void test_num_rings() { EXPECT_EQ(this->range().num_rings(), 0); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 0); }

  void test_multipolygon_it()
  {
    rmm::device_uvector<vec_2d<T>> d_points = this->copy_leading_point_multipolygon();
    rmm::device_uvector<vec_2d<T>> expected(0, this->stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_points, expected);
  }

  void test_point_it()
  {
    rmm::device_uvector<vec_2d<T>> d_points = this->copy_all_points();
    rmm::device_uvector<vec_2d<T>> expected(0, this->stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_points, expected);
  }

  void test_geometry_offsets_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_geometry_offsets();
    auto expected                              = make_device_vector<std::size_t>({0});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_part_offset_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_part_offsets();
    auto expected                              = make_device_vector<std::size_t>({0});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_ring_offset_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_ring_offsets();
    auto expected                              = make_device_vector<std::size_t>({0});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_ring_idx_from_point_idx()
  {
    rmm::device_uvector<std::size_t> d_ring_idx = this->copy_ring_idx_from_point_idx();
    auto expected                               = make_device_vector<std::size_t>({});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_ring_idx, expected);
  }

  void test_part_idx_from_ring_idx()
  {
    rmm::device_uvector<std::size_t> d_part_idx = this->copy_part_idx_from_ring_idx();
    auto expected                               = make_device_vector<std::size_t>({});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_part_idx, expected);
  }

  void test_geometry_idx_from_part_idx()
  {
    rmm::device_uvector<std::size_t> d_geometry_idx = this->copy_geometry_idx_from_part_idx();
    auto expected                                   = make_device_vector<std::size_t>({});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_geometry_idx, expected);
  }

  void test_array_access_operator()
  {
    // Nothing to access
    SUCCEED();
  }

  void test_multipolygon_point_count_it()
  {
    rmm::device_uvector<std::size_t> d_point_count = this->copy_multipolygon_point_count();
    rmm::device_uvector<std::size_t> expected(0, this->stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_point_count, expected);
  }

  void test_multipolygon_ring_count_it()
  {
    rmm::device_uvector<std::size_t> d_ring_count = this->copy_multipolygon_ring_count();
    rmm::device_uvector<std::size_t> expected(0, this->stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_ring_count, expected);
  }
};

TYPED_TEST_CASE(MultipolygonRangeEmptyTest, FloatingPointTypes);

TYPED_TEST(MultipolygonRangeEmptyTest, EmptyMultipolygonRange) { this->run_test(); }

template <typename T>
class MultipolygonRangeOneTest : public MultipolygonRangeTestBase<T> {
  void make_test_multipolygon()
  {
    auto geometry_offsets = make_device_vector<std::size_t>({0, 2});
    auto part_offsets     = make_device_vector<std::size_t>({0, 1, 2});
    auto ring_offsets     = make_device_vector<std::size_t>({0, 4, 8});
    auto coordinates      = make_device_vector<vec_2d<T>>(
      {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}});

    this->test_multipolygon = std::make_unique<multipolygon_array<rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<vec_2d<T>>>>(
      std::move(geometry_offsets),
      std::move(part_offsets),
      std::move(ring_offsets),
      std::move(coordinates));
  }

  void test_num_multipolygons() { EXPECT_EQ(this->range().num_multipolygons(), 1); }

  void test_num_polygons() { EXPECT_EQ(this->range().num_polygons(), 2); }

  void test_num_rings() { EXPECT_EQ(this->range().num_rings(), 2); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 8); }

  void test_multipolygon_it()
  {
    rmm::device_uvector<vec_2d<T>> d_points = this->copy_leading_point_multipolygon();
    auto expected                           = make_device_vector<vec_2d<T>>({{0, 0}});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_points, expected);
  }

  void test_point_it()
  {
    rmm::device_uvector<vec_2d<T>> d_points = this->copy_all_points();
    auto expected                           = make_device_vector<vec_2d<T>>(
      {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_points, expected);
  }

  void test_geometry_offsets_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_geometry_offsets();
    auto expected                              = make_device_vector<std::size_t>({0, 2});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_part_offset_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_part_offsets();
    auto expected                              = make_device_vector<std::size_t>({0, 1, 2});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_ring_offset_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_ring_offsets();
    auto expected                              = make_device_vector<std::size_t>({0, 4, 8});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_ring_idx_from_point_idx()
  {
    rmm::device_uvector<std::size_t> d_ring_idx = this->copy_ring_idx_from_point_idx();
    auto expected = make_device_vector<std::size_t>({0, 0, 0, 0, 1, 1, 1, 1});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_ring_idx, expected);
  }

  void test_part_idx_from_ring_idx()
  {
    rmm::device_uvector<std::size_t> d_part_idx = this->copy_part_idx_from_ring_idx();
    auto expected                               = make_device_vector<std::size_t>({0, 1});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_part_idx, expected);
  }

  void test_geometry_idx_from_part_idx()
  {
    rmm::device_uvector<std::size_t> d_geometry_idx = this->copy_geometry_idx_from_part_idx();
    auto expected                                   = make_device_vector<std::size_t>({0, 0});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_geometry_idx, expected);
  }

  void test_array_access_operator()
  {
    auto all_points = this->copy_all_points_of_ith_multipolygon(0);
    auto expected   = make_device_vector<vec_2d<T>>(
      {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {10, 10}, {11, 10}, {11, 11}, {10, 10}});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(all_points, expected);
  }

  void test_multipolygon_point_count_it()
  {
    rmm::device_uvector<std::size_t> d_point_count = this->copy_multipolygon_point_count();
    auto expected                                  = make_device_vector<std::size_t>({8});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_point_count, expected);
  }

  void test_multipolygon_ring_count_it()
  {
    rmm::device_uvector<std::size_t> d_ring_count = this->copy_multipolygon_ring_count();
    auto expected                                 = make_device_vector<std::size_t>({2});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_ring_count, expected);
  }
};

TYPED_TEST_CASE(MultipolygonRangeOneTest, FloatingPointTypes);

TYPED_TEST(MultipolygonRangeOneTest, OneMultipolygonRange) { this->run_test(); }

template <typename T>
class MultipolygonRangeOneThousandTest : public MultipolygonRangeTestBase<T> {
 public:
  struct make_points_functor {
    __device__ auto operator()(std::size_t i)
    {
      auto geometry_idx    = i / 4;
      auto intra_point_idx = i % 4;
      return vec_2d<T>{geometry_idx * T{10.} + intra_point_idx,
                       geometry_idx * T{10.} + intra_point_idx};
    }
  };

  void make_test_multipolygon()
  {
    auto geometry_offsets = rmm::device_vector<std::size_t>(1001);
    auto part_offsets     = rmm::device_vector<std::size_t>(1001);
    auto ring_offsets     = rmm::device_vector<std::size_t>(1001);
    auto coordinates      = rmm::device_vector<vec_2d<T>>(4000);

    thrust::sequence(
      rmm::exec_policy(this->stream()), geometry_offsets.begin(), geometry_offsets.end());

    thrust::sequence(rmm::exec_policy(this->stream()), part_offsets.begin(), part_offsets.end());

    thrust::sequence(
      rmm::exec_policy(this->stream()), ring_offsets.begin(), ring_offsets.end(), 0, 4);

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     coordinates.begin(),
                     coordinates.end(),
                     make_points_functor{});

    this->test_multipolygon = std::make_unique<multipolygon_array<rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<std::size_t>,
                                                                  rmm::device_vector<vec_2d<T>>>>(
      std::move(geometry_offsets),
      std::move(part_offsets),
      std::move(ring_offsets),
      std::move(coordinates));
  }

  void test_num_multipolygons() { EXPECT_EQ(this->range().num_multipolygons(), 1000); }

  void test_num_polygons() { EXPECT_EQ(this->range().num_polygons(), 1000); }

  void test_num_rings() { EXPECT_EQ(this->range().num_rings(), 1000); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 4000); }

  void test_multipolygon_it()
  {
    rmm::device_uvector<vec_2d<T>> d_points = this->copy_leading_point_multipolygon();
    rmm::device_uvector<vec_2d<T>> expected(1000, this->stream());
    thrust::tabulate(rmm::exec_policy(this->stream()),
                     expected.begin(),
                     expected.end(),
                     [] __device__(std::size_t i) {
                       return vec_2d<T>{i * T{10.}, i * T{10.}};
                     });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_points, expected);
  }

  void test_point_it()
  {
    rmm::device_uvector<vec_2d<T>> d_points = this->copy_all_points();
    rmm::device_uvector<vec_2d<T>> expected(4000, this->stream());

    thrust::tabulate(
      rmm::exec_policy(this->stream()), expected.begin(), expected.end(), make_points_functor{});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_points, expected);
  }

  void test_geometry_offsets_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_geometry_offsets();
    auto expected = rmm::device_uvector<std::size_t>(1001, this->stream());

    thrust::sequence(rmm::exec_policy(this->stream()), expected.begin(), expected.end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_part_offset_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_part_offsets();
    auto expected = rmm::device_uvector<std::size_t>(1001, this->stream());

    thrust::sequence(rmm::exec_policy(this->stream()), expected.begin(), expected.end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_ring_offset_it()
  {
    rmm::device_uvector<std::size_t> d_offsets = this->copy_ring_offsets();
    auto expected = rmm::device_uvector<std::size_t>(1001, this->stream());

    thrust::sequence(rmm::exec_policy(this->stream()), expected.begin(), expected.end(), 0, 4);

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_offsets, expected);
  }

  void test_ring_idx_from_point_idx()
  {
    rmm::device_uvector<std::size_t> d_ring_idx = this->copy_ring_idx_from_point_idx();
    auto expected = rmm::device_uvector<std::size_t>(4000, this->stream());

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     expected.begin(),
                     expected.end(),
                     [] __device__(std::size_t i) { return i / 4; });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_ring_idx, expected);
  }

  void test_part_idx_from_ring_idx()
  {
    rmm::device_uvector<std::size_t> d_part_idx = this->copy_part_idx_from_ring_idx();
    auto expected = rmm::device_uvector<std::size_t>(1000, this->stream());

    thrust::sequence(rmm::exec_policy(this->stream()), expected.begin(), expected.end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_part_idx, expected);
  }

  void test_geometry_idx_from_part_idx()
  {
    rmm::device_uvector<std::size_t> d_geometry_idx = this->copy_geometry_idx_from_part_idx();
    auto expected = rmm::device_uvector<std::size_t>(1000, this->stream());

    thrust::sequence(rmm::exec_policy(this->stream()), expected.begin(), expected.end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_geometry_idx, expected);
  }

  void test_array_access_operator()
  {
    auto all_points = this->copy_all_points_of_ith_multipolygon(777);
    auto expected   = make_device_vector<vec_2d<T>>({
      {7770, 7770},
      {7771, 7771},
      {7772, 7772},
      {7773, 7773},
    });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(all_points, expected);
  }

  void test_multipolygon_point_count_it()
  {
    rmm::device_uvector<std::size_t> d_point_count = this->copy_multipolygon_point_count();
    auto expected = rmm::device_uvector<std::size_t>(1000, this->stream());

    thrust::fill(rmm::exec_policy(this->stream()), expected.begin(), expected.end(), 4);

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_point_count, expected);
  }

  void test_multipolygon_ring_count_it()
  {
    rmm::device_uvector<std::size_t> d_ring_count = this->copy_multipolygon_ring_count();
    auto expected = rmm::device_uvector<std::size_t>(1000, this->stream());

    thrust::fill(rmm::exec_policy(this->stream()), expected.begin(), expected.end(), 1);

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_ring_count, expected);
  }
};

TYPED_TEST_CASE(MultipolygonRangeOneThousandTest, FloatingPointTypes);

TYPED_TEST(MultipolygonRangeOneThousandTest, OneThousandMultipolygonRange) { this->run_test(); }
