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

#include <cuspatial/distance/point_distance.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/vec_2d.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace cuspatial {

using namespace cudf;
using namespace cudf::test;

template <typename T>
struct PairwisePointDistanceTest : public ::testing::Test {
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwisePointDistanceTest, TestTypes);

TYPED_TEST(PairwisePointDistanceTest, Empty)
{
  using T = TypeParam;

  auto x1 = fixed_width_column_wrapper<T>{};
  auto y1 = fixed_width_column_wrapper<T>{};
  auto x2 = fixed_width_column_wrapper<T>{};
  auto y2 = fixed_width_column_wrapper<T>{};

  auto expect = fixed_width_column_wrapper<T>{};

  auto got = pairwise_point_distance(x1, y1, x2, y2);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

struct PairwisePointDistanceTestThrow : public ::testing::Test {
};

TEST_F(PairwisePointDistanceTestThrow, SizeMismatch)
{
  auto x1 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y1 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto x2 = fixed_width_column_wrapper<float>{};
  auto y2 = fixed_width_column_wrapper<float>{};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_distance(x1, y1, x2, y2), cuspatial::logic_error);
}

TEST_F(PairwisePointDistanceTestThrow, SizeMismatch2)
{
  auto x1 = fixed_width_column_wrapper<float>{};
  auto y1 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto x2 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y2 = fixed_width_column_wrapper<float>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_distance(x1, y1, x2, y2), cuspatial::logic_error);
}

TEST_F(PairwisePointDistanceTestThrow, TypeMismatch)
{
  auto x1 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y1 = fixed_width_column_wrapper<double>{1, 2, 3};
  auto x2 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y2 = fixed_width_column_wrapper<float>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_distance(x1, y1, x2, y2), cuspatial::logic_error);
}

TEST_F(PairwisePointDistanceTestThrow, TypeMismatch2)
{
  auto x1 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y1 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto x2 = fixed_width_column_wrapper<double>{1, 2, 3};
  auto y2 = fixed_width_column_wrapper<double>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_distance(x1, y1, x2, y2), cuspatial::logic_error);
}

TEST_F(PairwisePointDistanceTestThrow, ContainsNull)
{
  auto x1 = fixed_width_column_wrapper<float>{{1, 2, 3}, {1, 0, 1}};
  auto y1 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto x2 = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y2 = fixed_width_column_wrapper<float>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_distance(x1, y1, x2, y2), cuspatial::logic_error);
}

}  // namespace cuspatial
