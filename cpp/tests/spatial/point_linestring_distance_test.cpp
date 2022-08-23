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

TYPED_TEST(PairwisePointLinestringDistanceTest, EmptySingleComponent)
{
  using T = TypeParam;

  auto xy      = fixed_width_column_wrapper<T>{};
  auto offset  = fixed_width_column_wrapper<int32_t>{0};
  auto line_xy = fixed_width_column_wrapper<T>{};

  auto expect = fixed_width_column_wrapper<T>{};
  auto got    = pairwise_point_linestring_distance(
    std::nullopt, xy, std::nullopt, column_view(offset), line_xy);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwisePointLinestringDistanceTest, EmptyMultiComponent)
{
  using T = TypeParam;

  auto multipoint_offset      = fixed_width_column_wrapper<int32_t>{0};
  auto xy                     = fixed_width_column_wrapper<T>{};
  auto multilinestring_offset = fixed_width_column_wrapper<int32_t>{0};
  auto offset                 = fixed_width_column_wrapper<int32_t>{0};
  auto line_xy                = fixed_width_column_wrapper<T>{};

  auto expect = fixed_width_column_wrapper<T>{};
  auto got    = pairwise_point_linestring_distance(column_view(multipoint_offset),
                                                xy,
                                                column_view(multilinestring_offset),
                                                column_view(offset),
                                                line_xy);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwisePointLinestringDistanceTest, OnePairMultiPointMultiLinestring)
{
  using T = TypeParam;

  auto multipoint_offset      = fixed_width_column_wrapper<int32_t>{0, 2};
  auto xy                     = fixed_width_column_wrapper<T>{0.0, 0.0, 0.5, 0.5};
  auto multilinestring_offset = fixed_width_column_wrapper<int32_t>{0, 1};
  auto offset                 = fixed_width_column_wrapper<int32_t>{0, 2};
  auto line_xy                = fixed_width_column_wrapper<T>{1.0, 0.0, 0.0, 1.0};

  auto expect = fixed_width_column_wrapper<T>{0.0};
  auto got    = pairwise_point_linestring_distance(column_view(multipoint_offset),
                                                xy,
                                                column_view(multilinestring_offset),
                                                column_view(offset),
                                                line_xy);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

struct PairwisePointLinestringDistanceTestThrow : public ::testing::Test {
};

TEST_F(PairwisePointLinestringDistanceTestThrow, PointTypeMismatch)
{
  auto xy      = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3};
  auto offset  = fixed_width_column_wrapper<int32_t>{0, 6};
  auto line_xy = fixed_width_column_wrapper<double>{1, 1, 2, 2, 3, 3};

  EXPECT_THROW(pairwise_point_linestring_distance(
                 std::nullopt, xy, std::nullopt, column_view(offset), line_xy),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringDistanceTestThrow, ContainsNull)
{
  auto xy      = fixed_width_column_wrapper<float>{{1, 1, 2, 2, 3, 3}, {1, 0, 1, 1, 1, 1}};
  auto offset  = fixed_width_column_wrapper<int32_t>{0, 6};
  auto line_xy = fixed_width_column_wrapper<float>{1, 2, 3, 1, 2, 3};

  EXPECT_THROW(pairwise_point_linestring_distance(
                 std::nullopt, xy, std::nullopt, column_view(offset), line_xy),
               cuspatial::logic_error);
}

}  // namespace cuspatial
