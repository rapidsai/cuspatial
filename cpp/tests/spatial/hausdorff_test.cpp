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

#include <cuspatial/detail/hausdorff.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/hausdorff.hpp>

#include <thrust/iterator/constant_iterator.h>

using namespace cudf;
using namespace test;

template <typename T>
using hausdorff_acc = cuspatial::detail::hausdorff_acc<T>;

template <typename T>
hausdorff_acc<T> make_hausdorff_acc(thrust::pair<int32_t, int32_t> key,
                                    int32_t result_idx,
                                    int32_t col,
                                    T distance)
{
  return hausdorff_acc<T>{key, result_idx, col, col, distance, distance, 0};
}

template <typename T>
hausdorff_acc<T> make_hausdorff_acc(thrust::pair<int32_t, int32_t> key,
                                    int32_t result_idx,
                                    int32_t col_l,
                                    int32_t col_r,
                                    T min_l,
                                    T min_r,
                                    T max)
{
  return hausdorff_acc<T>{key, result_idx, col_l, col_r, min_l, min_r, max};
}

template <typename T>
void expect_haus_eq(hausdorff_acc<T> const& a, hausdorff_acc<T> const& b)
{
  using namespace cuspatial::detail;

  EXPECT_EQ(a.key, b.key);
  EXPECT_EQ(a.result_idx, b.result_idx);
  EXPECT_EQ(a.col_l, b.col_l);
  EXPECT_EQ(a.col_r, b.col_r);
  EXPECT_EQ(a.min_l, b.min_l);
  EXPECT_EQ(a.min_r, b.min_r);
  EXPECT_EQ(a.max, b.max);
}

template <typename T>
struct HausdorffTest : public BaseFixture {
};

using TestTypes = Types<double>;

TYPED_TEST_CASE(HausdorffTest, TestTypes);

TYPED_TEST(HausdorffTest, Binop1)
{
  using T = TypeParam;

  auto key   = thrust::make_pair<int64_t, int64_t>(0, 0);
  auto dst   = static_cast<int64_t>(0);
  auto col_a = static_cast<int64_t>(0);
  auto col_b = static_cast<int64_t>(1);

  auto a = make_hausdorff_acc<T>(key, dst, col_a, static_cast<int64_t>(5));
  auto b = make_hausdorff_acc<T>(key, dst, col_b, static_cast<int64_t>(7));

  auto expected = make_hausdorff_acc<T>(key, dst, col_a, col_b, 5, 7, 0);

  expect_haus_eq(a + b, expected);
}

TYPED_TEST(HausdorffTest, Binop2)
{
  using T = TypeParam;

  auto key   = thrust::make_pair<int64_t, int64_t>(0, 0);
  auto dst   = static_cast<int64_t>(0);
  auto col_0 = static_cast<int64_t>(0);
  auto col_1 = static_cast<int64_t>(1);

  auto a = make_hausdorff_acc<T>(key, dst, col_0, 3.6);
  auto b = make_hausdorff_acc<T>(key, dst, col_0, 8.2);
  auto c = make_hausdorff_acc<T>(key, dst, col_0, 1.4);

  auto d = make_hausdorff_acc<T>(key, dst, col_1, 8.4);
  auto e = make_hausdorff_acc<T>(key, dst, col_1, 5.3);
  auto f = make_hausdorff_acc<T>(key, dst, col_1, 5.0);

  auto expected = make_hausdorff_acc<T>(key, dst, col_0, col_1, 1.4, 5, 0);

  auto result = (a + b) + ((c + d) + (e + f));

  expect_haus_eq(result, expected);
}

TYPED_TEST(HausdorffTest, Empty)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, SingleTrajectorySinglePoint)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({152.2});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({2351.0});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, TwoShortSpaces)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({0, 5, 4});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({0, 12, 3});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0, 5, 13, 0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, TwoShortSpaces2)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({1, 5, 4, 2, 3, 7});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({1, 12, 3, 8, 4, 7});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 3, 4});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0.0,
                                                             7.0710678118654755,
                                                             5.3851648071345037,
                                                             5.0000000000000000,
                                                             0.0,
                                                             4.1231056256176606,
                                                             5.0,
                                                             5.0990195135927854,
                                                             0.0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, 10kSpaces10Points)
{
  using T = TypeParam;

  constexpr cudf::size_type num_spaces         = 10000;
  constexpr cudf::size_type elements_per_space = 10;
  constexpr cudf::size_type num_elements       = elements_per_space * num_spaces;

  auto zero_iter         = thrust::make_constant_iterator<T>(0);
  auto counting_iter     = thrust::make_counting_iterator<cudf::size_type>(0);
  auto space_offset_iter = thrust::make_transform_iterator(
    counting_iter, [elements_per_space](auto idx) { return idx * elements_per_space; });

  auto x = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
  auto y = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    space_offset_iter, space_offset_iter + num_spaces);

  auto expected = cudf::test::fixed_width_column_wrapper<T>(
    zero_iter, zero_iter + ((int64_t)num_spaces * (int64_t)num_spaces));

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view());
}

TYPED_TEST(HausdorffTest, 10Spaces10kPoints)
{
  using T = TypeParam;

  constexpr cudf::size_type num_spaces         = 10;
  constexpr cudf::size_type elements_per_space = 80000;
  constexpr cudf::size_type num_elements       = elements_per_space * num_spaces;

  auto zero_iter         = thrust::make_constant_iterator<T>(0);
  auto counting_iter     = thrust::make_counting_iterator<cudf::size_type>(0);
  auto space_offset_iter = thrust::make_transform_iterator(
    counting_iter, [elements_per_space](auto idx) { return idx * elements_per_space; });

  auto x = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
  auto y = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    space_offset_iter, space_offset_iter + num_spaces);

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + (num_spaces * num_spaces));

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view());
}

TYPED_TEST(HausdorffTest, MoreSpacesThanPoints)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({0});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({0});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1});

  EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, space_offsets), cuspatial::logic_error);
}

TYPED_TEST(HausdorffTest, TooFewPoints)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({0});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({0});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1});

  EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, space_offsets), cuspatial::logic_error);
}

TYPED_TEST(HausdorffTest, ThreeSpacesLengths543)
{
  using T = TypeParam;

  auto x = cudf::test::fixed_width_column_wrapper<T>(
    {0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0});
  auto y = cudf::test::fixed_width_column_wrapper<T>(
    {1.0, 2.0, 3.0, 5.0, 7.0, 0.0, 2.0, 3.0, 6.0, 1.0, 3.0, 6.0});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 5, 9});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({
    0.0000000000000000,
    4.1231056256176606,
    4.0000000000000000,
    3.6055512754639896,
    0.0000000000000000,
    1.4142135623730951,
    4.4721359549995796,
    1.4142135623730951,
    0.0000000000000000,
  });

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}
