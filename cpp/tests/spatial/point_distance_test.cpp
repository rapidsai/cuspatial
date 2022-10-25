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

TYPED_TEST(PairwisePointDistanceTest, SingleToSingleEmpty)
{
  using T = TypeParam;

  auto offset1 = std::nullopt;
  auto offset2 = std::nullopt;

  auto xy1 = fixed_width_column_wrapper<T>{};
  auto xy2 = fixed_width_column_wrapper<T>{};

  auto expect = fixed_width_column_wrapper<T>{};

  auto got = pairwise_point_distance(offset1, xy1, offset2, xy2);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwisePointDistanceTest, SingleToMultiEmpty)
{
  using T = TypeParam;

  auto offset1        = std::nullopt;
  column_view offset2 = fixed_width_column_wrapper<cudf::size_type>{0};

  auto xy1 = fixed_width_column_wrapper<T>{};
  auto xy2 = fixed_width_column_wrapper<T>{};

  auto expect = fixed_width_column_wrapper<T>{};

  auto got = pairwise_point_distance(offset1, xy1, offset2, xy2);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwisePointDistanceTest, MultiToSingleEmpty)
{
  using T = TypeParam;

  column_view offset1 = fixed_width_column_wrapper<cudf::size_type>{0};
  auto offset2        = std::nullopt;

  auto xy1 = fixed_width_column_wrapper<T>{};
  auto xy2 = fixed_width_column_wrapper<T>{};

  auto expect = fixed_width_column_wrapper<T>{};

  auto got = pairwise_point_distance(offset1, xy1, offset2, xy2);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwisePointDistanceTest, MultiToMultiEmpty)
{
  using T = TypeParam;

  column_view offset1 = fixed_width_column_wrapper<cudf::size_type>{0};
  column_view offset2 = fixed_width_column_wrapper<cudf::size_type>{0};

  auto xy1 = fixed_width_column_wrapper<T>{};
  auto xy2 = fixed_width_column_wrapper<T>{};

  auto expect = fixed_width_column_wrapper<T>{};

  auto got = pairwise_point_distance(offset1, xy1, offset2, xy2);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

struct PairwisePointDistanceTestThrow : public ::testing::Test {
};

TEST_F(PairwisePointDistanceTestThrow, SizeMismatch)
{
  column_view offset1 = fixed_width_column_wrapper<cudf::size_type>{0, 3};
  column_view offset2 = fixed_width_column_wrapper<cudf::size_type>{0};

  auto xy1 = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3};
  auto xy2 = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_distance(offset1, xy1, offset2, xy2), cuspatial::logic_error);
}

TEST_F(PairwisePointDistanceTestThrow, SizeMismatch2)
{
  column_view offset1 = fixed_width_column_wrapper<cudf::size_type>{0, 3};
  auto offset2        = std::nullopt;

  auto xy1 = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3};
  auto xy2 = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_distance(offset1, xy1, offset2, xy2), cuspatial::logic_error);
}

TEST_F(PairwisePointDistanceTestThrow, TypeMismatch)
{
  auto offset1 = std::nullopt;
  auto offset2 = std::nullopt;
  auto xy1     = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3};
  auto xy2     = fixed_width_column_wrapper<double>{1, 1, 2, 2, 3, 3};

  EXPECT_THROW(pairwise_point_distance(offset1, xy1, offset2, xy2), cuspatial::logic_error);
}
}  // namespace cuspatial
