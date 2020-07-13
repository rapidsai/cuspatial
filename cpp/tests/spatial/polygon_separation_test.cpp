/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <vector>

#include <cuspatial/error.hpp>
#include <cuspatial/hausdorff.hpp>
#include <cuspatial/polygon_separation.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include "gtest/gtest.h"

#include <thrust/iterator/constant_iterator.h>

using namespace cudf;
using namespace test;

template <typename T>
struct MinimumEuclideanDistanceTest : public BaseFixture {
};

using TestTypes = Types<double>;

TYPED_TEST_CASE(MinimumEuclideanDistanceTest, TestTypes);

TYPED_TEST(MinimumEuclideanDistanceTest, ZeroShapes)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({});

  auto actual = cuspatial::directed_polygon_separation(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(MinimumEuclideanDistanceTest, TwoShapesEdgeToPoint)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({-2, 0, -2, 1, 3, +1});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({+1, 0, -1, 1, 0, -1});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 3});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0.0, 1.4142135623730951, 1.0, 0.0});

  auto actual = cuspatial::directed_polygon_separation(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(MinimumEuclideanDistanceTest, TwoShapesPointToPoint)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({-1, -2, -3, 1, 2, 3});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({-1, -3, -2, 1, 3, 2});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 3});

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>({0.0, 2.8284271247461903, 2.8284271247461903, 0.0});

  auto actual = cuspatial::directed_polygon_separation(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(MinimumEuclideanDistanceTest, InvalidTypeTest)
{
  EXPECT_TRUE(false);  // todo
}

TYPED_TEST(MinimumEuclideanDistanceTest, MismatchedTypeTest)
{
  EXPECT_TRUE(false);  // todo
}

TYPED_TEST(MinimumEuclideanDistanceTest, EdgesOnly)
{
  EXPECT_TRUE(false);  // todo
}

TYPED_TEST(MinimumEuclideanDistanceTest, PointsOnly)
{
  EXPECT_TRUE(false);  // todo
}
