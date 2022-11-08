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
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/linestring_distance.cuh>
#include <cuspatial/experimental/ranges/multilinestring_range.cuh>
#include <cuspatial/vec_2d.hpp>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct PairwiseLinestringDistanceTest : public ::testing::Test {
};

struct PairwiseLinestringDistanceTestUntyped : public ::testing::Test {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PairwiseLinestringDistanceTest, TestTypes);

TYPED_TEST(PairwiseLinestringDistanceTest, FromSeparateArrayInputs)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto constexpr num_pairs = 1;

  auto a_cart2d = rmm::device_vector<vec_2d<T>>{
    CartVec({{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}})};
  auto b_cart2d = rmm::device_vector<vec_2d<T>>{
    CartVec({{0.0f, 1.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}, {3.0f, 1.0f}, {4.0f, 1.0f}})};
  auto offset = make_device_vector<int32_t>({0, 5});

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset.size() - 1,
                                                  offset.begin(),
                                                  a_cart2d.size(),
                                                  a_cart2d.begin());
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset.size() - 1,
                                                  offset.begin(),
                                                  b_cart2d.size(),
                                                  b_cart2d.begin());

  auto got      = rmm::device_vector<T>(num_pairs);
  auto expected = rmm::device_vector<T>{std::vector<T>{1.0}};

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FromSamePointArrayInput)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto constexpr num_pairs = 1;

  auto cart2ds = make_device_vector<vec_2d<T>>(
    {{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}});
  auto offset_a = make_device_vector<int32_t>({0, 3});
  auto offset_b = make_device_vector<int32_t>({0, 4});

  auto got      = rmm::device_vector<T>(1);
  auto expected = make_device_vector<T>({0.0});

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset_a.size() - 1,
                                                  offset_a.begin(),
                                                  3,
                                                  cart2ds.begin());

  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset_b.size() - 1,
                                                  offset_b.begin(),
                                                  4,
                                                  thrust::next(cart2ds.begin()));

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FromTransformIterator)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto constexpr num_pairs = 1;

  auto a_cart2d_x = rmm::device_vector<T>{std::vector<T>{0.0, 1.0, 2.0, 3.0, 4.0}};
  auto a_cart2d_y = rmm::device_vector<T>(5, 0.0);

  auto a_begin = make_vec_2d_iterator(a_cart2d_x.begin(), a_cart2d_y.begin());

  auto b_cart2d_x = rmm::device_vector<T>{std::vector<T>{0.0, 1.0, 2.0, 3.0, 4.0}};
  auto b_cart2d_y = rmm::device_vector<T>(5, 1.0);

  auto b_begin = make_vec_2d_iterator(b_cart2d_x.begin(), b_cart2d_y.begin());

  auto offset = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 5}};

  auto got      = rmm::device_vector<T>{1};
  auto expected = rmm::device_vector<T>{std::vector<T>{1.0}};

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset.size() - 1,
                                                  offset.begin(),
                                                  a_cart2d_x.size(),
                                                  a_begin);

  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset.size() - 1,
                                                  offset.begin(),
                                                  b_cart2d_x.size(),
                                                  b_begin);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FromMixedIterator)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto constexpr num_pairs = 1;

  auto a_cart2d = rmm::device_vector<vec_2d<T>>{
    CartVec({{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}})};

  auto b_cart2d_x = rmm::device_vector<T>{std::vector<T>{0.0, 1.0, 2.0, 3.0, 4.0}};
  auto b_cart2d_y = rmm::device_vector<T>(5, 1.0);

  auto b_begin = make_vec_2d_iterator(b_cart2d_x.begin(), b_cart2d_y.begin());

  auto offset_a = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 5}};
  auto offset_b = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 5}};

  auto got      = rmm::device_vector<T>{1};
  auto expected = rmm::device_vector<T>{std::vector<T>{1.0}};

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset_a.size() - 1,
                                                  offset_a.begin(),
                                                  a_cart2d.size(),
                                                  a_cart2d.begin());

  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset_b.size() - 1,
                                                  offset_b.begin(),
                                                  b_cart2d_x.size(),
                                                  b_begin);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FromLongInputs)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto constexpr num_pairs  = 5;
  auto constexpr num_points = 1000;

  auto a_cart2d_x_begin = thrust::make_constant_iterator(T{0.0});
  auto a_cart2d_y_begin = thrust::make_counting_iterator(T{0.0});
  auto a_cart2d_begin   = make_vec_2d_iterator(a_cart2d_x_begin, a_cart2d_y_begin);

  auto b_cart2d_x_begin = thrust::make_constant_iterator(T{42.0});
  auto b_cart2d_y_begin = thrust::make_counting_iterator(T{0.0});
  auto b_cart2d_begin   = make_vec_2d_iterator(b_cart2d_x_begin, b_cart2d_y_begin);

  auto offset =
    rmm::device_vector<int32_t>{std::vector<int32_t>{0, 100, 200, 300, 400, num_points}};

  auto got      = rmm::device_vector<T>{num_pairs};
  auto expected = rmm::device_vector<T>{std::vector<T>{42.0, 42.0, 42.0, 42.0, 42.0}};

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset.size() - 1,
                                                  offset.begin(),
                                                  num_points,
                                                  a_cart2d_begin);

  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  offset.size() - 1,
                                                  offset.begin(),
                                                  num_points,
                                                  b_cart2d_begin);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringParallel)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (1.0, 0.0), (2.0, 1.0)

  int32_t constexpr num_pairs = 1;

  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 1.0, 1.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<T>({1.0, 0.0, 2.0, 1.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({0.7071067811865476});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringEndpointsDistance)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0), (2.0, 2.0)
  // Linestring 2: (2.0, 0.0), (1.0, -1.0), (0.0, -1.0)

  auto constexpr num_pairs = 1;

  auto linestring1_offsets   = make_device_vector<int32_t>({0, 3});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 1.0, 1.0, 2.0, 2.0});
  auto linestring2_offsets   = make_device_vector<T>({0, 3});
  auto linestring2_points_xy = make_device_vector<T>({2.0, 0.0, 1.0, -1.0, 0.0, -1.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({1.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringProjectionNotOnLine)
{
  using T = TypeParam;

  auto constexpr num_pairs = 1;

  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (3.0, 1.5), (3.0, 2.0)
  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 1.0, 1.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<T>({3.0, 1.5, 3.0, 2.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({2.0615528128088303});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringPerpendicular)
{
  using T = TypeParam;

  auto constexpr num_pairs = 1;

  // Linestring 1: (0.0, 0.0), (2.0, 0.0)
  // Linestring 2: (1.0, 1.0), (1.0, 2.0)
  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 2.0, 0.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<T>({1.0, 1.0, 1.0, 2.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({1.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringIntersects)
{
  using T = TypeParam;

  auto constexpr num_pairs = 1;

  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (0.0, 1.0), (1.0, 0.0)
  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 1.0, 1.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<T>({0.0, 1.0, 1.0, 0.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({0.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringSharedVertex)
{
  using T = TypeParam;

  auto constexpr num_pairs = 1;

  // Linestring 1: (0.0, 0.0), (0.0, 2.0), (2.0, 2.0)
  // Linestring 2: (2.0, 2.0), (2.0, 1.0), (1.0, 1.0), (2.5, 0.0)
  auto linestring1_offsets   = make_device_vector<int32_t>({0, 3});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 0.0, 2.0, 2.0, 2.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 4});
  auto linestring2_points_xy = make_device_vector<T>({2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.5, 0.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({0.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringCoincide)
{
  using T = TypeParam;

  auto constexpr num_pairs = 1;
  // Linestring 1: (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
  // Linestring 2: (2.0, 1.0), (1.0, 1.0), (1.0, 0.0), (2.0, 0.0), (2.0, 0.5)
  auto linestring1_offsets   = make_device_vector<int32_t>({0, 4});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 5});
  auto linestring2_points_xy =
    make_device_vector<T>({2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.5});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({0.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairDegenerateCollinearNoIntersect)
{
  using T = TypeParam;

  auto constexpr num_pairs = 1;

  // Linestring1: (0.0, 0.0) -> (0.0, 1.0)
  // Linestring2: (0.0, 2.0) -> (0.0, 3.0)
  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 0.0, 1.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<T>({0.0, 2.0, 0.0, 3.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({1.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairCollinearNoIntersect)
{
  using T = TypeParam;

  auto constexpr num_pairs = 1;

  // Linestring1: (0.0, 0.0) -> (1.0, 1.0)
  // Linestring2: (2.0, 2.0) -> (3.0, 3.0)
  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 1.0, 1.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<T>({2.0, 2.0, 3.0, 3.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({1.4142135623730951});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairDegenerateCollinearIntersect)
{
  using T = TypeParam;

  auto constexpr num_pairs = 1;

  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<T>({0.0, 0.0, 2.0, 2.0});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<T>({1.0, 1.0, 3.0, 3.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({0.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TEST_F(PairwiseLinestringDistanceTestUntyped, OnePairDeterminantDoublePrecisionDenormalized)
{
  // Vector ab: (1e-155, 2e-155)
  // Vector cd: (2e-155, 1e-155)
  // determinant of matrix [a, b] = -3e-310, a denormalized number

  auto constexpr num_pairs = 1;

  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<double>({0.0, 0.0, 1e-155, 2e-155});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<double>({4e-155, 5e-155, 6e-155, 6e-155});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<double>({4.24264068711929e-155});
  auto got      = rmm::device_vector<double>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TEST_F(PairwiseLinestringDistanceTestUntyped, OnePairDeterminantSinglePrecisionDenormalized)
{
  // Vector ab: (1e-20, 2e-20)
  // Vector cd: (2e-20, 1e-20)
  // determinant of matrix [ab, cd] = -3e-40, a denormalized number

  auto constexpr num_pairs = 1;

  auto linestring1_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = make_device_vector<float>({0.0, 0.0, 1e-20, 2e-20});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<float>({4e-20, 5e-20, 6e-20, 6e-20});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<float>({4.2426405524813e-20});
  auto got      = rmm::device_vector<float>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  // Expect a slightly greater floating point error compared to 4 ULP
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got, 1e-25f);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairRandom1)
{
  using T = TypeParam;

  auto constexpr num_pairs   = 1;
  auto linestring1_offsets   = make_device_vector<int32_t>({0, 3});
  auto linestring1_points_xy = make_device_vector<T>({-22556.235212018168,
                                                      41094.0501840996,
                                                      -16375.655690574613,
                                                      42992.319790050366,
                                                      -20082.724633593425,
                                                      33759.13529113619});
  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = make_device_vector<T>(
    {4365.496374409238, -59857.47177852941, 1671.0269165650761, -54931.9723439855});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({91319.97744223749});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairIntersectFromRealData1)
{
  // Example extracted from a pair of trajectry in geolife dataset
  using T                  = TypeParam;
  auto constexpr num_pairs = 1;

  auto linestring1_offsets   = make_device_vector<int32_t>({0, 5});
  auto linestring1_points_xy = make_device_vector<T>({39.97551667,
                                                      116.33028333,
                                                      39.97585,
                                                      116.3304,
                                                      39.97598333,
                                                      116.33046667,
                                                      39.9761,
                                                      116.3305,
                                                      39.97623333,
                                                      116.33056667});

  auto linestring2_offsets   = make_device_vector<int32_t>({0, 55});
  auto linestring2_points_xy = make_device_vector<T>(
    {39.97381667, 116.34211667, 39.97341667, 116.34215,    39.9731,     116.34218333,
     39.97293333, 116.34221667, 39.97233333, 116.34225,    39.97218333, 116.34243333,
     39.97218333, 116.34296667, 39.97215,    116.34478333, 39.97168333, 116.34486667,
     39.97093333, 116.34485,    39.97073333, 116.34468333, 39.9705,     116.34461667,
     39.96991667, 116.34465,    39.96961667, 116.34465,    39.96918333, 116.34466667,
     39.96891667, 116.34465,    39.97531667, 116.33036667, 39.97533333, 116.32961667,
     39.97535,    116.3292,     39.97515,    116.32903333, 39.97506667, 116.32985,
     39.97508333, 116.33128333, 39.9751,     116.33195,    39.97513333, 116.33618333,
     39.97511667, 116.33668333, 39.97503333, 116.33818333, 39.97513333, 116.34,
     39.97523333, 116.34045,    39.97521667, 116.34183333, 39.97503333, 116.342,
     39.97463333, 116.34203333, 39.97443333, 116.3422,     39.96838333, 116.3445,
     39.96808333, 116.34451667, 39.96771667, 116.3445,     39.96745,    116.34453333,
     39.96735,    116.34493333, 39.9673,     116.34506667, 39.96718333, 116.3451,
     39.96751667, 116.34483333, 39.9678,     116.3448,     39.9676,     116.3449,
     39.96741667, 116.345,      39.9672,     116.34506667, 39.97646667, 116.33006667,
     39.9764,     116.33015,    39.97625,    116.33026667, 39.9762,     116.33038333,
     39.97603333, 116.33036667, 39.97581667, 116.3303,     39.9757,     116.33033333,
     39.97551667, 116.33035,    39.97535,    116.3304,     39.97543333, 116.33078333,
     39.97538333, 116.33066667});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({0.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairFromRealData)
{
  // Example extracted from a pair of trajectry in geolife dataset
  using T = TypeParam;

  auto constexpr num_pairs = 1;
  auto linestring1_offsets = make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy =
    make_device_vector<T>({39.9752666666667, 116.334316666667, 39.9752666666667, 116.334533333333});
  auto linestring2_offsets = make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy =
    make_device_vector<T>({39.9752666666667, 116.323966666667, 39.9752666666667, 116.3236});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  // Steps to reproduce:
  // Create a float32/float64 numpy array with the literal inputs as above.
  // Construct a shapely.geometry.LineString object and compute the result.
  // Cast the result to np.float32/np.float64.
  auto expected =
    make_device_vector<T>({std::is_same_v<T, float> ? 0.010353088f : 0.010349999999988313});
  auto got = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, TwoPairsSingleLineString)
{
  using T                  = TypeParam;
  auto constexpr num_pairs = 2;

  // First pair lhs:
  // (41658.902315589876, 14694.11814724456)->(46600.70359801489, 8771.431887804214)->
  // (47079.510547637154, 10199.68027155776)->(51498.48049880379, 17049.62665643919)
  // First pair rhs:
  // (24046.170375947084, 27878.56737867571)->(20614.007047185743, 26489.74880629428)

  // Second pair lhs:
  // (-27429.917796286478, -33240.8339287343)->(-21764.269974046114, -37974.45515744517)->
  // (-14460.71813363161, -31333.481529957502)->(-18226.13032712476, -30181.03842467982)
  // Second pair rhs:
  // (48381.39607717942, -8366.313156569413)->(53346.77764665915, -2066.3869793077383)

  auto linestring1_offsets   = make_device_vector<int32_t>({0, 4, 8});
  auto linestring1_points_xy = make_device_vector<T>({41658.902315589876,
                                                      14694.11814724456,
                                                      46600.70359801489,
                                                      8771.431887804214,
                                                      47079.510547637154,
                                                      10199.68027155776,
                                                      51498.48049880379,
                                                      17049.62665643919,
                                                      -27429.917796286478,
                                                      -33240.8339287343,
                                                      -21764.269974046114,
                                                      -37974.45515744517,
                                                      -14460.71813363161,
                                                      -31333.481529957502,
                                                      -18226.13032712476,
                                                      -30181.03842467982});

  auto linestring2_offsets   = make_device_vector<int32_t>({0, 2, 4});
  auto linestring2_points_xy = make_device_vector<T>({
    24046.170375947084,
    27878.56737867571,
    20614.007047185743,
    26489.74880629428,
    48381.39607717942,
    -8366.313156569413,
    53346.77764665915,
    -2066.3869793077383,
  });

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({22000.86425379464, 66907.56415814416});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FourPairsSingleLineString)
{
  using T                  = TypeParam;
  auto constexpr num_pairs = 4;

  auto linestring1_offsets = make_device_vector<int32_t>({0, 3, 5, 8, 10});

  auto linestring1_points_xy =
    make_device_vector<T>({0, 1, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 2, 2, -2, 0, 2, 2, -2, -2});

  auto linestring2_offsets   = make_device_vector<int32_t>({0, 4, 7, 9, 12});
  auto linestring2_points_xy = make_device_vector<T>(
    {1, 1, 2, 1, 2, 0, 3, 0, 1, 0, 1, 1, 1, 2, 2, 0, 0, 2, 1, 1, 5, 5, 10, 0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = make_device_vector<T>({static_cast<T>(std::sqrt(2.0) * 0.5), 1.0, 0.0, 0.0});
  auto got      = rmm::device_vector<T>(expected.size());

  auto mlinestrings1 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring1_offsets.size() - 1,
                                                  linestring1_offsets.begin(),
                                                  linestring1_points_xy.size() / 2,
                                                  linestring1_points_it);
  auto mlinestrings2 = make_multilinestring_range(num_pairs,
                                                  thrust::make_counting_iterator(0),
                                                  linestring2_offsets.size() - 1,
                                                  linestring2_offsets.begin(),
                                                  linestring2_points_xy.size() / 2,
                                                  linestring2_points_it);

  auto ret = pairwise_linestring_distance(mlinestrings1, mlinestrings2, got.begin());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}
