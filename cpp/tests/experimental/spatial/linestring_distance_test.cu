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

#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cuspatial {
namespace test {

template <typename T>
struct PairwiseLinestringDistanceTest : public ::testing::Test {
};

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

}  // namespace test
}  // namespace cuspatial
