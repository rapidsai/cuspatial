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

#include "tests/utility/vector_equality.hpp"

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/linestring_distance.cuh>
#include <cuspatial/vec_2d.hpp>

#include <initializer_list>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cuspatial {
namespace test {

using namespace cudf;

template <typename T>
struct PairwiseLinestringDistanceTest : public ::testing::Test {
  template <typename U>
  auto make_device_vector(std::initializer_list<U> inl)
  {
    return rmm::device_vector<U>(inl.begin(), inl.end());
  }
};

struct PairwiseLinestringDistanceTestUntyped : public ::testing::Test {};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PairwiseLinestringDistanceTest, TestTypes);

TYPED_TEST(PairwiseLinestringDistanceTest, FromSeparateArrayInputs)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto a_cart2d = rmm::device_vector<vec_2d<T>>{
    CartVec({{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}})};
  auto b_cart2d = rmm::device_vector<vec_2d<T>>{
    CartVec({{0.0f, 1.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}, {3.0f, 1.0f}, {4.0f, 1.0f}})};
  auto offset = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 5}};

  auto distance = rmm::device_vector<T>{1};
  auto expected = rmm::device_vector<T>{std::vector<T>{1.0}};

  auto ret = pairwise_linestring_distance(offset.begin(),
                                          offset.end(),
                                          a_cart2d.begin(),
                                          a_cart2d.end(),
                                          offset.begin(),
                                          b_cart2d.begin(),
                                          b_cart2d.end(),
                                          distance.begin());

  test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(offset.size() - 1, std::distance(distance.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FromSamePointArrayInput)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto cart2ds = rmm::device_vector<vec_2d<T>>{
    CartVec({{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}})};
  auto offset_a = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 3}};
  auto offset_b = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 4}};

  auto a_begin = cart2ds.begin();
  auto a_end   = cart2ds.begin() + 3;
  auto b_begin = cart2ds.begin() + 1;
  auto b_end   = cart2ds.end();

  auto distance = rmm::device_vector<T>{1};
  auto expected = rmm::device_vector<T>{std::vector<T>{0.0}};

  auto ret = pairwise_linestring_distance(offset_a.begin(),
                                          offset_a.end(),
                                          a_begin,
                                          a_end,
                                          offset_a.begin(),
                                          b_begin,
                                          b_end,
                                          distance.begin());

  test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(offset_a.size() - 1, std::distance(distance.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FromTransformIterator)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto a_cart2d_x = rmm::device_vector<T>{std::vector<T>{0.0, 1.0, 2.0, 3.0, 4.0}};
  auto a_cart2d_y = rmm::device_vector<T>(5, 0.0);

  auto a_begin = make_vec_2d_iterator(a_cart2d_x.begin(), a_cart2d_y.begin());
  auto a_end   = a_begin + a_cart2d_x.size();

  auto b_cart2d_x = rmm::device_vector<T>{std::vector<T>{0.0, 1.0, 2.0, 3.0, 4.0}};
  auto b_cart2d_y = rmm::device_vector<T>(5, 1.0);

  auto b_begin = make_vec_2d_iterator(b_cart2d_x.begin(), b_cart2d_y.begin());
  auto b_end   = b_begin + b_cart2d_x.size();

  auto offset = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 5}};

  auto distance = rmm::device_vector<T>{1};
  auto expected = rmm::device_vector<T>{std::vector<T>{1.0}};

  auto ret = pairwise_linestring_distance(
    offset.begin(), offset.end(), a_begin, a_end, offset.begin(), b_begin, b_end, distance.begin());

  test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(offset.size() - 1, std::distance(distance.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FromMixedIterator)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto a_cart2d = rmm::device_vector<vec_2d<T>>{
    CartVec({{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}})};

  auto b_cart2d_x = rmm::device_vector<T>{std::vector<T>{0.0, 1.0, 2.0, 3.0, 4.0}};
  auto b_cart2d_y = rmm::device_vector<T>(5, 1.0);

  auto b_begin = make_vec_2d_iterator(b_cart2d_x.begin(), b_cart2d_y.begin());
  auto b_end   = b_begin + b_cart2d_x.size();

  auto offset_a = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 5}};
  auto offset_b = rmm::device_vector<int32_t>{std::vector<int32_t>{0, 5}};

  auto distance = rmm::device_vector<T>{1};
  auto expected = rmm::device_vector<T>{std::vector<T>{1.0}};

  auto ret = pairwise_linestring_distance(offset_a.begin(),
                                          offset_a.end(),
                                          a_cart2d.begin(),
                                          a_cart2d.end(),
                                          offset_b.begin(),
                                          b_begin,
                                          b_end,
                                          distance.begin());

  test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(offset_a.size() - 1, std::distance(distance.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, FromLongInputs)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_points = 1000;

  auto a_cart2d_x_begin = thrust::make_constant_iterator(T{0.0});
  auto a_cart2d_y_begin = thrust::make_counting_iterator(T{0.0});
  auto a_cart2d_begin   = make_vec_2d_iterator(a_cart2d_x_begin, a_cart2d_y_begin);
  auto a_cart2d_end     = a_cart2d_begin + num_points;

  auto b_cart2d_x_begin = thrust::make_constant_iterator(T{42.0});
  auto b_cart2d_y_begin = thrust::make_counting_iterator(T{0.0});
  auto b_cart2d_begin   = make_vec_2d_iterator(b_cart2d_x_begin, b_cart2d_y_begin);
  auto b_cart2d_end     = b_cart2d_begin + num_points;

  auto offset =
    rmm::device_vector<int32_t>{std::vector<int32_t>{0, 100, 200, 300, 400, num_points}};

  auto distance = rmm::device_vector<T>{5};
  auto expected = rmm::device_vector<T>{std::vector<T>{42.0, 42.0, 42.0, 42.0, 42.0}};

  auto ret = pairwise_linestring_distance(offset.begin(),
                                          offset.end(),
                                          a_cart2d_begin,
                                          a_cart2d_end,
                                          offset.begin(),
                                          b_cart2d_begin,
                                          b_cart2d_end,
                                          distance.begin());

  test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(offset.size() - 1, std::distance(distance.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringParallel)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (1.0, 0.0), (2.0, 1.0)

  int32_t constexpr num_pairs = 1;

  auto linestring1_offsets   = this->template make_device_vector<int32_t>({0, 2});
  auto linestring1_points_xy = this->template make_device_vector<T>({0.0, 0.0, 1.0, 1.0});
  auto linestring2_offsets   = this->template make_device_vector<int32_t>({0, 2});
  auto linestring2_points_xy = this->template make_device_vector<T>({1.0, 0.0, 2.0, 1.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = this->template make_device_vector({0.7071067811865476});
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

  expect_columns_equivalent(expected, *got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringEndpointsDistance)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0), (2.0, 2.0)
  // Linestring 2: (2.0, 0.0), (1.0, -1.0), (0.0, -1.0)

  auto constexpr num_pairs = 1;

  auto linestring1_offsets   = this->template make_device_vector<int32_t>({0, 3});
  auto linestring1_points_xy = this->template make_device_vector<T>({0.0, 0.0, 1.0, 1.0, 2.0, 2.0});
  auto linestring2_offsets   = this->template make_device_vector<T>({0, 3});
  auto linestring2_points_xy =
    this->template make_device_vector<T>({2.0, 0.0, 1.0, -1.0, 0.0, -1.0});

  auto linestring1_points_it = make_vec_2d_iterator(linestring1_points_xy.begin());
  auto linestring2_points_it = make_vec_2d_iterator(linestring2_points_xy.begin());

  auto expected = this->template make_device_vector<T>({1.0});
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

  expect_columns_equivalent(expected, *got);
  EXPECT_EQ(num_pairs, std::distance(got.begin(), ret));
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringProjectionNotOnLine)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (3.0, 1.5), (3.0, 2.0)
  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<T> linestring1_points_x{0.0, 1.0};
  wrapper<T> linestring1_points_y{0.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{3.0, 3.0};
  wrapper<T> linestring2_points_y{1.5, 2.0};

  wrapper<T> expected{2.0615528128088303};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringPerpendicular)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (2.0, 0.0)
  // Linestring 2: (1.0, 1.0), (1.0, 2.0)
  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<T> linestring1_points_x{0.0, 2.0};
  wrapper<T> linestring1_points_y{0.0, 0.0};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{1.0, 1.0};
  wrapper<T> linestring2_points_y{1.0, 2.0};

  wrapper<T> expected{1.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringIntersects)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (0.0, 1.0), (1.0, 0.0)
  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<T> linestring1_points_x{0.0, 1.0};
  wrapper<T> linestring1_points_y{0.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{0.0, 1.0};
  wrapper<T> linestring2_points_y{1.0, 0.0};

  wrapper<T> expected{0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringSharedVertex)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (0.0, 2.0), (2.0, 2.0)
  // Linestring 2: (2.0, 2.0), (2.0, 1.0), (1.0, 1.0), (2.5, 0.0)
  wrapper<cudf::size_type> linestring1_offsets{0, 3};
  wrapper<T> linestring1_points_x{0.0, 0.0, 2.0};
  wrapper<T> linestring1_points_y{0.0, 2.0, 2.0};
  wrapper<cudf::size_type> linestring2_offsets{0, 4};
  wrapper<T> linestring2_points_x{2.0, 2.0, 1.0, 2.5};
  wrapper<T> linestring2_points_y{2.0, 1.0, 1.0, 0.0};

  wrapper<T> expected{0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringCoincide)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
  // Linestring 2: (2.0, 1.0), (1.0, 1.0), (1.0, 0.0), (2.0, 0.0), (2.0, 0.5)
  wrapper<cudf::size_type> linestring1_offsets{0, 4};
  wrapper<T> linestring1_points_x{0.0, 1.0, 1.0, 0.0};
  wrapper<T> linestring1_points_y{0.0, 0.0, 1.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0, 5};
  wrapper<T> linestring2_points_x{2.0, 1.0, 1.0, 2.0, 2.0};
  wrapper<T> linestring2_points_y{1.0, 1.0, 0.0, 0.0, 0.5};

  wrapper<T> expected{0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairDegenerateCollinearNoIntersect)
{
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<T> linestring1_points_x{0.0, 0.0};
  wrapper<T> linestring1_points_y{0.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{0.0, 0.0};
  wrapper<T> linestring2_points_y{2.0, 3.0};

  wrapper<T> expected{1.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairCollinearNoIntersect)
{
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<T> linestring1_points_x{0.0, 1.0};
  wrapper<T> linestring1_points_y{0.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{2.0, 3.0};
  wrapper<T> linestring2_points_y{2.0, 3.0};

  wrapper<T> expected{1.4142135623730951};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairDegenerateCollinearIntersect)
{
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<T> linestring1_points_x{0.0, 2.0};
  wrapper<T> linestring1_points_y{0.0, 2.0};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{1.0, 3.0};
  wrapper<T> linestring2_points_y{1.0, 3.0};

  wrapper<T> expected{0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, OnePairDeterminantDoublePrecisionDenormalized)
{
  // Vector ab: (1e-155, 2e-155)
  // Vector cd: (2e-155, 1e-155)
  // determinant of matrix [a, b] = -3e-310, a denormalized number

  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<double> linestring1_points_x{0.0, 1e-155};
  wrapper<double> linestring1_points_y{0.0, 2e-155};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<double> linestring2_points_x{4e-155, 6e-155};
  wrapper<double> linestring2_points_y{5e-155, 6e-155};

  wrapper<double> expected{4.24264068711929e-155};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, OnePairDeterminantSinglePrecisionDenormalized)
{
  // Vector ab: (1e-20, 2e-20)
  // Vector cd: (2e-20, 1e-20)
  // determinant of matrix [ab, cd] = -3e-40, a denormalized number

  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<float> linestring1_points_x{0.0, 1e-20};
  wrapper<float> linestring1_points_y{0.0, 2e-20};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<float> linestring2_points_x{4e-20, 6e-20};
  wrapper<float> linestring2_points_y{5e-20, 6e-20};

  wrapper<float> expected{4.2426405524813e-20};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairRandom1)
{
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 3};
  wrapper<T> linestring1_points_x{-22556.235212018168, -16375.655690574613, -20082.724633593425};
  wrapper<T> linestring1_points_y{41094.0501840996, 42992.319790050366, 33759.13529113619};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{4365.496374409238, 1671.0269165650761};
  wrapper<T> linestring2_points_y{-59857.47177852941, -54931.9723439855};

  wrapper<T> expected{91319.97744223749};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairIntersectFromRealData1)
{
  // Example extracted from a pair of trajectry in geolife dataset
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 5};
  wrapper<T> linestring1_points_x{39.97551667, 39.97585, 39.97598333, 39.9761, 39.97623333};
  wrapper<T> linestring1_points_y{116.33028333, 116.3304, 116.33046667, 116.3305, 116.33056667};
  wrapper<cudf::size_type> linestring2_offsets{0, 55};
  wrapper<T> linestring2_points_x{
    39.97381667, 39.97341667, 39.9731,     39.97293333, 39.97233333, 39.97218333, 39.97218333,
    39.97215,    39.97168333, 39.97093333, 39.97073333, 39.9705,     39.96991667, 39.96961667,
    39.96918333, 39.96891667, 39.97531667, 39.97533333, 39.97535,    39.97515,    39.97506667,
    39.97508333, 39.9751,     39.97513333, 39.97511667, 39.97503333, 39.97513333, 39.97523333,
    39.97521667, 39.97503333, 39.97463333, 39.97443333, 39.96838333, 39.96808333, 39.96771667,
    39.96745,    39.96735,    39.9673,     39.96718333, 39.96751667, 39.9678,     39.9676,
    39.96741667, 39.9672,     39.97646667, 39.9764,     39.97625,    39.9762,     39.97603333,
    39.97581667, 39.9757,     39.97551667, 39.97535,    39.97543333, 39.97538333};
  wrapper<T> linestring2_points_y{
    116.34211667, 116.34215,    116.34218333, 116.34221667, 116.34225,    116.34243333,
    116.34296667, 116.34478333, 116.34486667, 116.34485,    116.34468333, 116.34461667,
    116.34465,    116.34465,    116.34466667, 116.34465,    116.33036667, 116.32961667,
    116.3292,     116.32903333, 116.32985,    116.33128333, 116.33195,    116.33618333,
    116.33668333, 116.33818333, 116.34,       116.34045,    116.34183333, 116.342,
    116.34203333, 116.3422,     116.3445,     116.34451667, 116.3445,     116.34453333,
    116.34493333, 116.34506667, 116.3451,     116.34483333, 116.3448,     116.3449,
    116.345,      116.34506667, 116.33006667, 116.33015,    116.33026667, 116.33038333,
    116.33036667, 116.3303,     116.33033333, 116.33035,    116.3304,     116.33078333,
    116.33066667};

  wrapper<T> expected{0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairFromRealData)
{
  // Example extracted from a pair of trajectry in geolife dataset
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 2};
  wrapper<T> linestring1_points_x{39.9752666666667, 39.9752666666667};
  wrapper<T> linestring1_points_y{116.334316666667, 116.334533333333};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{39.9752666666667, 39.9752666666667};
  wrapper<T> linestring2_points_y{116.323966666667, 116.3236};

  wrapper<T> expected =
    std::is_same_v<T, float> ? wrapper<T>{0.01035308837890625} : wrapper<T>{0.010349999999988313};
  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, TwoPairs)
{
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 4, 8};
  wrapper<T> linestring1_points_x{
    41658.902315589876,
    46600.70359801489,
    47079.510547637154,
    51498.48049880379,
    -27429.917796286478,
    -21764.269974046114,
    -14460.71813363161,
    -18226.13032712476,
  };
  wrapper<T> linestring1_points_y{14694.11814724456,
                                  8771.431887804214,
                                  10199.68027155776,
                                  17049.62665643919,
                                  -33240.8339287343,
                                  -37974.45515744517,
                                  -31333.481529957502,
                                  -30181.03842467982};
  wrapper<cudf::size_type> linestring2_offsets{0, 2, 4};
  wrapper<T> linestring2_points_x{
    24046.170375947084,
    20614.007047185743,
    48381.39607717942,
    53346.77764665915,
  };
  wrapper<T> linestring2_points_y{
    27878.56737867571,
    26489.74880629428,
    -8366.313156569413,
    -2066.3869793077383,
  };

  wrapper<T> expected{22000.86425379464, 66907.56415814416};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, FourPairs)
{
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 3, 5, 8, 10};
  wrapper<T> linestring1_points_x{0, 1, -1, 0, 0, 0, 2, -2, 2, -2};
  wrapper<T> linestring1_points_y{1, 0, 0, 0, 1, 0, 2, 0, 2, -2};
  wrapper<cudf::size_type> linestring2_offsets{0, 4, 7, 9, 12};
  wrapper<T> linestring2_points_x{1, 2, 2, 3, 1, 1, 1, 2, 0, 1, 5, 10};
  wrapper<T> linestring2_points_y{1, 1, 0, 0, 0, 1, 2, 0, 2, 1, 5, 0};

  wrapper<T> expected{std::sqrt(2.0) * 0.5, 1.0, 0.0, 0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

}  // namespace test
}  // namespace cuspatial
