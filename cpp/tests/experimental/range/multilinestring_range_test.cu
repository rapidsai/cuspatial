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

#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct MultilinestringRangeTest : public BaseFixture {
  void run_segment_test_single(std::initializer_list<std::size_t> geometry_offset,
                               std::initializer_list<std::size_t> part_offset,
                               std::initializer_list<vec_2d<T>> coordinates,
                               std::initializer_list<segment<T>> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);

    auto rng = multilinestring_array.range();

    rmm::device_uvector<segment<T>> got(rng.num_segments(), stream());
    thrust::copy(rmm::exec_policy(stream()), rng.segment_begin(), rng.segment_end(), got.begin());

    auto d_expected = thrust::device_vector<segment<T>>(expected.begin(), expected.end());
    CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_point_count_test(std::initializer_list<std::size_t> geometry_offset,
                                            std::initializer_list<std::size_t> part_offset,
                                            std::initializer_list<vec_2d<T>> coordinates,
                                            std::initializer_list<std::size_t> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);

    auto d_expected = thrust::device_vector<std::size_t>(expected.begin(), expected.end());

    auto rng = multilinestring_array.range();

    rmm::device_uvector<std::size_t> got(rng.num_multilinestrings(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multilinestring_point_count_begin(),
                 rng.multilinestring_point_count_end(),
                 got.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_segment_count_test(std::initializer_list<std::size_t> geometry_offset,
                                              std::initializer_list<std::size_t> part_offset,
                                              std::initializer_list<vec_2d<T>> coordinates,
                                              std::initializer_list<std::size_t> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);

    auto d_expected = thrust::device_vector<std::size_t>(expected.begin(), expected.end());

    auto rng = multilinestring_array.range();

    rmm::device_uvector<std::size_t> got(rng.num_multilinestrings(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multilinestring_segment_count_begin(),
                 rng.multilinestring_segment_count_end(),
                 got.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_linestring_count_test(std::initializer_list<std::size_t> geometry_offset,
                                                 std::initializer_list<std::size_t> part_offset,
                                                 std::initializer_list<vec_2d<T>> coordinates,
                                                 std::initializer_list<std::size_t> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);

    auto d_expected = thrust::device_vector<std::size_t>(expected.begin(), expected.end());

    auto rng = multilinestring_array.range();

    rmm::device_uvector<std::size_t> got(rng.num_multilinestrings(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multilinestring_linestring_count_begin(),
                 rng.multilinestring_linestring_count_end(),
                 got.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_as_multipoint_test(
    std::initializer_list<std::size_t> geometry_offset,
    std::initializer_list<std::size_t> part_offset,
    std::initializer_list<vec_2d<T>> coordinates,
    std::initializer_list<std::initializer_list<vec_2d<T>>> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);
    auto multilinestring_range = multilinestring_array.range();

    auto multipoint_range = multilinestring_range.as_multipoint_range();

    thrust::device_vector<std::size_t> got_geometry_offset(multipoint_range.offsets_begin(),
                                                           multipoint_range.offsets_end());
    thrust::device_vector<vec_2d<T>> got_coordinates(multipoint_range.point_begin(),
                                                     multipoint_range.point_end());

    auto expected_multipoint = make_multipoints_array(expected);
    auto expected_range      = expected_multipoint.range();

    thrust::device_vector<std::size_t> expected_geometry_offset(expected_range.offsets_begin(),
                                                                expected_range.offsets_end());
    thrust::device_vector<vec_2d<T>> expected_coordinates(expected_range.point_begin(),
                                                          expected_range.point_end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_geometry_offset, got_geometry_offset);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_coordinates, got_coordinates);
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(MultilinestringRangeTest, TestTypes);

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorOneSegmentTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_segment_test_single, {0, 1}, {0, 2}, {P{0, 0}, P{1, 1}}, {S{P{0, 0}, P{1, 1}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorTwoSegmentTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_segment_test_single,
                     {0, 2},
                     {0, 2, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
                     {S{P{0, 0}, P{1, 1}}, S{P{2, 2}, P{3, 3}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorTwoSegmentTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_segment_test_single,
                     {0, 1, 2},
                     {0, 2, 4},
                     {P{0, 0}, P{1, 1}, P{0, 0}, P{1, 1}},
                     {S{P{0, 0}, P{1, 1}}, S{P{0, 0}, P{1, 1}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorManyPairTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_segment_test_single,
                     {0, 1, 2, 3},
                     {0, 6, 11, 14},
                     {P{0, 0},
                      P{1, 1},
                      P{2, 2},
                      P{3, 3},
                      P{4, 4},
                      P{5, 5},

                      P{10, 10},
                      P{11, 11},
                      P{12, 12},
                      P{13, 13},
                      P{14, 14},

                      P{20, 20},
                      P{21, 21},
                      P{22, 22}},

                     {S{P{0, 0}, P{1, 1}},
                      S{P{1, 1}, P{2, 2}},
                      S{P{2, 2}, P{3, 3}},
                      S{P{3, 3}, P{4, 4}},
                      S{P{4, 4}, P{5, 5}},
                      S{P{10, 10}, P{11, 11}},
                      S{P{11, 11}, P{12, 12}},
                      S{P{12, 12}, P{13, 13}},
                      S{P{13, 13}, P{14, 14}},
                      S{P{20, 20}, P{21, 21}},
                      S{P{21, 21}, P{22, 22}}});
}

/// FIXME: Currently, segment iterator doesn't handle empty linestrings.
TYPED_TEST(MultilinestringRangeTest, DISABLED_SegmentIteratorWithEmptyLineTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_segment_test_single,
    {0, 1, 2, 3},
    {0, 3, 3, 6},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
    {S{P{0, 0}, P{1, 1}}, S{P{1, 1}, P{2, 2}}, S{P{10, 10}, P{11, 11}}, S{P{11, 11}, P{12, 12}}});
}

TYPED_TEST(MultilinestringRangeTest, PerMultilinestringCountTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_point_count_test,
                     {0, 1, 2, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {3, 0, 3});
}

TYPED_TEST(MultilinestringRangeTest, PerMultilinestringCountTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_point_count_test,
                     {0, 1, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {3, 3});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_multilinestring_segment_count_test, {0, 1}, {0, 3}, {P{0, 0}, P{1, 1}, P{2, 2}}, {2});
}

// FIXME: contains empty linestring
TYPED_TEST(MultilinestringRangeTest, DISABLED_MultilinestringSegmentCountTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {2, 2});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest3)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 2},
                     {0, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {2, 2});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest4)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 3},
                     {0, 3, 5, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{20, 20}, P{21, 21}},
                     {2, 2});
}

// FIXME: contains empty linestring
TYPED_TEST(MultilinestringRangeTest, DISABLED_MultilinestringSegmentCountTest5)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 2, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {2, 0, 2});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringLinestringCountTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_multilinestring_linestring_count_test,
    {0, 1, 2, 3},
    {0, 3, 6, 9},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}, P{20, 20}, P{21, 21}, P{22, 22}},
    {1, 1, 1});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringLinestringCountTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_multilinestring_linestring_count_test,
    {0, 1, 3},
    {0, 3, 6, 9},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}, P{20, 20}, P{21, 21}, P{22, 22}},
    {1, 2});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 1},
                     {0, 3},
                     {P{0, 0}, P{1, 1}, P{2, 2}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 2},
                     {0, 3, 5},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{4, 4}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{4, 4}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest3)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 1, 2},
                     {0, 3, 5},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}}, {P{10, 10}, P{11, 11}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest4)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 1, 3},
                     {0, 3, 5, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}, P{13, 13}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}}, {P{10, 10}, P{11, 11}, P{12, 12}, P{13, 13}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest5)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 1},
                     {0, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{2, 3}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}, P{2, 3}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest6)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 2},
                     {0, 2, 4},
                     {P{1, 1}, P{0, 0}, P{6, 6}, P{6, 7}},
                     {{P{1, 1}, P{0, 0}, P{6, 6}, P{6, 7}}});
}
