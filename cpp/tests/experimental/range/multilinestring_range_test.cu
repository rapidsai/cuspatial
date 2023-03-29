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

  void run_tiled_segments_test_single(std::initializer_list<std::size_t> geometry_offset,
                                      std::initializer_list<std::size_t> part_offset,
                                      std::initializer_list<vec_2d<T>> coordinates,
                                      std::size_t length,
                                      std::initializer_list<segment<T>> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);

    auto rng = multilinestring_array.range();

    rmm::device_uvector<segment<T>> got(length, stream());
    auto it = rng.segment_tiled_begin();
    thrust::copy(rmm::exec_policy(stream()), it, it + length, got.begin());

    auto d_expected = thrust::device_vector<segment<T>>(expected.begin(), expected.end());
    CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(d_expected, got);
  }

  void run_per_multilinestring_point_count_test(std::initializer_list<std::size_t> geometry_offset,
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
                 rng.per_multilinestring_point_count_begin(),
                 rng.per_multilinestring_point_count_end(),
                 got.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
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

TYPED_TEST(MultilinestringRangeTest, TiledSegmentIteratorTestOneLine)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_tiled_segments_test_single,
                     {0, 1},
                     {0, 2},
                     {P{0, 0}, P{1, 1}},
                     3,
                     {S{P{0, 0}, P{1, 1}}, S{P{0, 0}, P{1, 1}}, S{P{0, 0}, P{1, 1}}});
}

TYPED_TEST(MultilinestringRangeTest, TiledSegmentIteratorTestTwoLines)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_tiled_segments_test_single,
                     {0, 1, 2},
                     {0, 2, 4},
                     {P{0, 0}, P{1, 1}, P{10, 10}, P{11, 11}},
                     5,
                     {
                       S{P{0, 0}, P{1, 1}},
                       S{P{10, 10}, P{11, 11}},
                       S{P{0, 0}, P{1, 1}},
                       S{P{10, 10}, P{11, 11}},
                       S{P{0, 0}, P{1, 1}},
                     });
}

TYPED_TEST(MultilinestringRangeTest, TiledSegmentIteratorTestThreeLines)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_tiled_segments_test_single,
    {0, 1, 2, 3},
    {0, 2, 4, 8},
    {P{0, 0}, P{1, 1}, P{10, 10}, P{11, 11}, P{20, 20}, P{21, 21}, P{22, 22}, P{23, 23}},
    10,
    {S{P{0, 0}, P{1, 1}},
     S{P{10, 10}, P{11, 11}},
     S{P{20, 20}, P{21, 21}},
     S{P{21, 21}, P{22, 22}},
     S{P{22, 22}, P{23, 23}},
     S{P{0, 0}, P{1, 1}},
     S{P{10, 10}, P{11, 11}},
     S{P{20, 20}, P{21, 21}},
     S{P{21, 21}, P{22, 22}},
     S{P{22, 22}, P{23, 23}}});
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

  CUSPATIAL_RUN_TEST(this->run_per_multilinestring_point_count_test,
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

  CUSPATIAL_RUN_TEST(this->run_per_multilinestring_point_count_test,
                     {0, 1, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {3, 3});
}
