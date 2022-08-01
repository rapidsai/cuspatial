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

#include <cuspatial/distance/point_linestring_distance.hpp>
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
struct PairwisePointLinestringDistanceTest : public ::testing::Test {
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwisePointLinestringDistanceTest, TestTypes);

TYPED_TEST(PairwisePointLinestringDistanceTest, Empty)
{
  using T = TypeParam;

  auto x      = fixed_width_column_wrapper<T>{};
  auto y      = fixed_width_column_wrapper<T>{};
  auto offset = fixed_width_column_wrapper<int32_t>{};
  auto line_x = fixed_width_column_wrapper<T>{};
  auto line_y = fixed_width_column_wrapper<T>{};

  auto expect = fixed_width_column_wrapper<T>{};

  auto got = pairwise_point_linestring_distance(x, y, offset, line_x, line_y);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

struct PairwisePointLinestringDistanceTestThrow : public ::testing::Test {
};

TEST_F(PairwisePointLinestringDistanceTestThrow, PointSizeMismatch)
{
  auto x      = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y      = fixed_width_column_wrapper<float>{};
  auto offset = fixed_width_column_wrapper<int32_t>{0};
  auto line_x = fixed_width_column_wrapper<float>{1, 2, 3};
  auto line_y = fixed_width_column_wrapper<float>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_linestring_distance(x, y, offset, line_x, line_y),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringDistanceTestThrow, LinestringSizeMismatch)
{
  auto x      = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y      = fixed_width_column_wrapper<float>{1, 2, 3};
  auto offset = fixed_width_column_wrapper<int32_t>{0};
  auto line_x = fixed_width_column_wrapper<float>{};
  auto line_y = fixed_width_column_wrapper<float>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_linestring_distance(x, y, offset, line_x, line_y),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringDistanceTestThrow, PointTypeMismatch)
{
  auto x      = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y      = fixed_width_column_wrapper<double>{1, 2, 3};
  auto offset = fixed_width_column_wrapper<int32_t>{0};
  auto line_x = fixed_width_column_wrapper<float>{1, 2, 3};
  auto line_y = fixed_width_column_wrapper<float>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_linestring_distance(x, y, offset, line_x, line_y),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringDistanceTestThrow, LinestringTypeMismatch)
{
  auto x      = fixed_width_column_wrapper<float>{1, 2, 3};
  auto y      = fixed_width_column_wrapper<float>{1, 2, 3};
  auto offset = fixed_width_column_wrapper<int32_t>{0};
  auto line_x = fixed_width_column_wrapper<double>{1, 2, 3};
  auto line_y = fixed_width_column_wrapper<float>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_linestring_distance(x, y, offset, line_x, line_y),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringDistanceTestThrow, ContainsNull)
{
  auto x      = fixed_width_column_wrapper<float>{{1, 2, 3}, {1, 0, 1}};
  auto y      = fixed_width_column_wrapper<float>{1, 2, 3};
  auto offset = fixed_width_column_wrapper<int32_t>{0};
  auto line_x = fixed_width_column_wrapper<float>{1, 2, 3};
  auto line_y = fixed_width_column_wrapper<float>{1, 2, 3};

  auto expect = fixed_width_column_wrapper<float>{};

  EXPECT_THROW(pairwise_point_linestring_distance(x, y, offset, line_x, line_y),
               cuspatial::logic_error);
}

}  // namespace cuspatial
