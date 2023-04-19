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
  void run_multipolygon_segment_iterator_single(std::initializer_list<std::size_t> geometry_offset,
                                                std::initializer_list<std::size_t> part_offset,
                                                std::initializer_list<std::size_t> ring_offset,
                                                std::initializer_list<vec_2d<T>> coordinates,
                                                std::initializer_list<segment<T>> expected)
  {
    auto multipolygon_array =
      make_multipolygon_array(geometry_offset, part_offset, ring_offset, coordinates);
    auto rng = multipolygon_array.range();

    auto got = rmm::device_uvector<segment<T>>(rng.num_segments(), stream());

    thrust::copy(rmm::exec_policy(stream()), rng.segment_begin(), rng.segment_end(), got.begin());

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

  void run_multipolygon_segment_count_single(
    std::initializer_list<std::size_t> geometry_offset,
    std::initializer_list<std::size_t> part_offset,
    std::initializer_list<std::size_t> ring_offset,
    std::initializer_list<vec_2d<T>> coordinates,
    std::initializer_list<std::size_t> expected_segment_counts)
  {
    auto multipolygon_array =
      make_multipolygon_array(geometry_offset, part_offset, ring_offset, coordinates);
    auto rng = multipolygon_array.range();

    auto got = rmm::device_uvector<std::size_t>(rng.num_multipolygons(), stream());

    thrust::copy(rmm::exec_policy(stream()),
                 rng.multipolygon_segment_count_begin(),
                 rng.multipolygon_segment_count_end(),
                 got.begin());

    auto d_expected = thrust::device_vector<std::size_t>(expected_segment_counts.begin(),
                                                         expected_segment_counts.end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(got, d_expected);
  }
};

TYPED_TEST_CASE(MultipolygonRangeTest, FloatingPointTypes);

TYPED_TEST(MultipolygonRangeTest, SegmentIterators)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_iterator_single,
                     {0, 1},
                     {0, 1},
                     {0, 4},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}},
                     {S{P{0, 0}, P{1, 0}}, S{P{1, 0}, P{1, 1}}, S{P{1, 1}, P{0, 0}}});
}

TYPED_TEST(MultipolygonRangeTest, SegmentIterators2)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_iterator_single,
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
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_iterator_single,
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
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_iterator_single,
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
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_count_single,
                     {0, 1},
                     {0, 1},
                     {0, 4},
                     {{0, 0}, {1, 0}, {1, 1}, {0, 0}},
                     {3});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonSegmentCount2)
{
  CUSPATIAL_RUN_TEST(
    this->run_multipolygon_segment_count_single,
    {0, 1},
    {0, 2},
    {0, 4, 8},
    {{0, 0}, {1, 0}, {1, 1}, {0, 0}, {0.2, 0.2}, {0.2, 0.3}, {0.3, 0.3}, {0.2, 0.2}},
    {6});
}

TYPED_TEST(MultipolygonRangeTest, MultipolygonSegmentCount3)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_count_single,
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
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_count_single,
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

// FIXME: multipolygon doesn't constructor doesn't allow empty rings, should it?
TYPED_TEST(MultipolygonRangeTest, DISABLED_MultipolygonSegmentCount_ConatainsEmptyRing)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_count_single,
                     {0, 2, 3},
                     {0, 2, 3, 4},
                     {0, 4, 4, 8, 12},
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

// FIXME: multipolygon doesn't constructor doesn't allow empty rings, should it?
TYPED_TEST(MultipolygonRangeTest, DISABLED_MultipolygonSegmentCount_ConatainsEmptyPart)
{
  CUSPATIAL_RUN_TEST(this->run_multipolygon_segment_count_single,
                     {0, 3, 4},
                     {0, 2, 2, 3, 4},
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
