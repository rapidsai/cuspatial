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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/hausdorff.hpp>

#include <thrust/iterator/constant_iterator.h>

using namespace cudf;
using namespace test;

template <typename T>
struct HausdorffTest : public BaseFixture {
};

using TestTypes = Types<double>;

TYPED_TEST_CASE(HausdorffTest, TestTypes);

TYPED_TEST(HausdorffTest, Empty)
{
  using T = TypeParam;

  auto x      = cudf::test::fixed_width_column_wrapper<T>({});
  auto y      = cudf::test::fixed_width_column_wrapper<T>({});
  auto spaces = cudf::test::fixed_width_column_wrapper<cudf::size_type>({});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, spaces);

  expect_columns_equal(expected, actual->view());
}

TYPED_TEST(HausdorffTest, SingleTrajectorySinglePoint)
{
  using T = TypeParam;

  auto x      = cudf::test::fixed_width_column_wrapper<T>({152.2});
  auto y      = cudf::test::fixed_width_column_wrapper<T>({2351.0});
  auto spaces = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, spaces);

  expect_columns_equal(expected, actual->view());
}

TYPED_TEST(HausdorffTest, TwoShortSpaces)
{
  using T = TypeParam;

  auto x      = cudf::test::fixed_width_column_wrapper<T>({0, 5, 4});
  auto y      = cudf::test::fixed_width_column_wrapper<T>({0, 12, 3});
  auto spaces = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1, 2});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0, 5, 13, 0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, spaces);

  expect_columns_equal(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, 10kSpacesSinglePoint)
{
  using T = TypeParam;

  constexpr cudf::size_type num_spaces              = 10000;
  constexpr cudf::size_type elements_per_trajectory = 1;

  auto zero_iter   = thrust::make_constant_iterator<T>(0);
  auto stride_iter = thrust::make_constant_iterator<cudf::size_type>(elements_per_trajectory);

  auto x = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_spaces);
  auto y = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_spaces);
  auto spaces =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(stride_iter, stride_iter + num_spaces);

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + (num_spaces * num_spaces));

  auto actual = cuspatial::directed_hausdorff_distance(x, y, spaces);

  expect_columns_equal(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, 2Spaces500kPoints)
{
  using T = TypeParam;

  constexpr cudf::size_type num_spaces              = 2;
  constexpr cudf::size_type elements_per_trajectory = 500000;

  auto zero_iter   = thrust::make_constant_iterator<T>(0);
  auto stride_iter = thrust::make_constant_iterator<cudf::size_type>(elements_per_trajectory);

  auto x = cudf::test::fixed_width_column_wrapper<T>(
    zero_iter, zero_iter + (elements_per_trajectory * num_spaces));
  auto y = cudf::test::fixed_width_column_wrapper<T>(
    zero_iter, zero_iter + (elements_per_trajectory * num_spaces));
  auto spaces =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(stride_iter, stride_iter + num_spaces);

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + (num_spaces * num_spaces));

  auto actual = cuspatial::directed_hausdorff_distance(x, y, spaces);

  expect_columns_equal(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, MoreSpacesThanPoints)
{
  using T = TypeParam;

  auto x      = cudf::test::fixed_width_column_wrapper<T>({0});
  auto y      = cudf::test::fixed_width_column_wrapper<T>({0});
  auto spaces = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1, 1});

  EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, spaces), cuspatial::logic_error);
}

TYPED_TEST(HausdorffTest, TooFewPoints)
{
  using T = TypeParam;

  auto x      = cudf::test::fixed_width_column_wrapper<T>({0});
  auto y      = cudf::test::fixed_width_column_wrapper<T>({0});
  auto spaces = cudf::test::fixed_width_column_wrapper<cudf::size_type>({2});

  // ideally this would throw, but we don't have a good way to catch the negative length.
  EXPECT_NO_THROW(cuspatial::directed_hausdorff_distance(x, y, spaces));
}

TYPED_TEST(HausdorffTest, SpaceWithNegativePointCount)
{
  using T = TypeParam;

  auto x      = cudf::test::fixed_width_column_wrapper<T>({0, 0});
  auto y      = cudf::test::fixed_width_column_wrapper<T>({0, 0});
  auto spaces = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1, -2});

  // ideally this would throw, but we don't have a good way to catch the negative length.
  EXPECT_NO_THROW(cuspatial::directed_hausdorff_distance(x, y, spaces));
}

TYPED_TEST(HausdorffTest, SpaceSizeZero)
{
  using T = TypeParam;

  auto x      = cudf::test::fixed_width_column_wrapper<T>({0});
  auto y      = cudf::test::fixed_width_column_wrapper<T>({0});
  auto spaces = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0});

  // ideally this would throw, but we don't have a good way to catch the zero length.
  EXPECT_NO_THROW(cuspatial::directed_hausdorff_distance(x, y, spaces));
}
