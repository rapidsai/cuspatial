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
#include <cuspatial/detail/hausdorff.cuh>

#include <thrust/iterator/constant_iterator.h>

using namespace cudf;
using namespace test;

template<typename T>
using haus = cuspatial::detail::haus<T>;

template<typename T>
void expect_haus_eq(haus<T> const& a, haus<T> const& b)
{
    using namespace cuspatial::detail;

    EXPECT_EQ(haus_key(a), haus_key(b));
    EXPECT_EQ(haus_dst(a), haus_dst(b));
    EXPECT_EQ(haus_col_l(a), haus_col_l(b));
    EXPECT_EQ(haus_col_r(a), haus_col_r(b));
    EXPECT_EQ(haus_min_l(a), haus_min_l(b));
    EXPECT_EQ(haus_min_r(a), haus_min_r(b));
    EXPECT_EQ(haus_max(a), haus_max(b));
    EXPECT_EQ(haus_res(a), haus_res(b));
}

template<typename T>
haus<T> make_haus(int32_t col, T value)
{
  return haus<T>{
    thrust::make_tuple(0, 0),
    0,
    col,
    col,
    value,
    value,
    0,
    value
  };
}

template <typename T>
struct HausdorffTest : public BaseFixture {
};

using TestTypes = Types<double>;

TYPED_TEST_CASE(HausdorffTest, TestTypes);

TYPED_TEST(HausdorffTest, Binop1)
{
    using T = TypeParam;

    auto op = cuspatial::detail::haus_reduce<T>{};

    auto a = make_haus<T>(0, 5);

    auto b = make_haus<T>(1, 7);

    auto expected = cuspatial::detail::haus<T>{
      thrust::make_tuple(0, 0), 0,
      0, 1,
      5, 7,
      0, 7
    };

    expect_haus_eq(op(a, b), expected);
}

TYPED_TEST(HausdorffTest, Binop2)
{
    using T = TypeParam;

    auto op = cuspatial::detail::haus_reduce<T>{};

    auto a = make_haus<T>(0, 3.6);
    auto b = make_haus<T>(0, 8.2);
    auto c = make_haus<T>(0, 1.4);

    auto d = make_haus<T>(1, 8.4);
    auto e = make_haus<T>(1, 5.3);
    auto f = make_haus<T>(1, 5.0);

    auto expected = cuspatial::detail::haus<T>{
      thrust::make_tuple(0, 0), 0,
      0, 1,
      1.4, 5,
      0, 5
    };

    auto result = op(op(a, b), op(op(c, d), op(e, f)));

    expect_haus_eq(result, expected);
}

TYPED_TEST(HausdorffTest, Empty)
{
    using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({});
    auto y = cudf::test::fixed_width_column_wrapper<T>({});
    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({});

    auto expected = cudf::test::fixed_width_column_wrapper<T>({});

    auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

    expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, SingleTrajectorySinglePoint)
{
    using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({  152.2 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 2351.0 });
    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0 });

    auto expected = cudf::test::fixed_width_column_wrapper<T>({ 0 });

    auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

    expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, TwoShortSpaces)
{
    using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({ 0, 5, 4 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 0, 12, 3 });
    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 1 });

    auto expected = cudf::test::fixed_width_column_wrapper<T>({ 0, 5, 13, 0 });

    auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

    expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, TwoShortSpaces2)
{
    using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({ 1,  5, 4, 2, 3, 7 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 1, 12, 3, 8, 4, 7 });
    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 3, 4 });

    auto expected = cudf::test::fixed_width_column_wrapper<T>({
        0.0, 7.0710678118654755, 5.3851648071345037, 5.0000000000000000, 0.0, 4.1231056256176606, 5.0, 5.0990195135927854, 0.0
    });

    auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

    expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(HausdorffTest, 10kSpaces10Points)
{
    using T = TypeParam;

    constexpr cudf::size_type num_spaces = 10000;
    constexpr cudf::size_type elements_per_space = 10;
    constexpr cudf::size_type num_elements = elements_per_space * num_spaces;

    auto zero_iter = thrust::make_constant_iterator<T>(0);
    auto counting_iter = thrust::make_counting_iterator<cudf::size_type>(0);
    auto space_offset_iter = thrust::make_transform_iterator(
        counting_iter,
        [elements_per_space] (auto idx) {
          return idx * elements_per_space;
        }
    );

    auto x = cudf::test::fixed_width_column_wrapper<T>(
      zero_iter,
      zero_iter + num_elements
    );

    auto y = cudf::test::fixed_width_column_wrapper<T>(
      zero_iter,
      zero_iter + num_elements
    );

    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      space_offset_iter,
      space_offset_iter + num_spaces
    );

    auto expected = cudf::test::fixed_width_column_wrapper<T>(
      zero_iter,
      zero_iter + ((int64_t) num_spaces * (int64_t) num_spaces)
    );

    auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

    expect_columns_equivalent(expected, actual->view());
}

TYPED_TEST(HausdorffTest, 10Spaces10kPoints)
{
    using T = TypeParam;

    constexpr cudf::size_type num_spaces = 10;
    constexpr cudf::size_type elements_per_space = 80000;
    constexpr cudf::size_type num_elements = elements_per_space * num_spaces;

    auto zero_iter = thrust::make_constant_iterator<T>(0);
    auto counting_iter = thrust::make_counting_iterator<cudf::size_type>(0);
    auto space_offset_iter = thrust::make_transform_iterator(
        counting_iter,
        [elements_per_space] (auto idx) {
          return idx * elements_per_space;
        }
    );

    auto x = cudf::test::fixed_width_column_wrapper<T>(
      zero_iter,
      zero_iter + num_elements
    );

    auto y = cudf::test::fixed_width_column_wrapper<T>(
      zero_iter,
      zero_iter + num_elements
    );

    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      space_offset_iter,
      space_offset_iter + num_spaces
    );

    auto expected = cudf::test::fixed_width_column_wrapper<T>(
      zero_iter,
      zero_iter + (num_spaces * num_spaces)
    );

    auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

    expect_columns_equivalent(expected, actual->view());
}

TYPED_TEST(HausdorffTest, MoreSpacesThanPoints)
{
    using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({ 0 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 0 });
    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 1 });

    EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, space_offsets), cuspatial::logic_error);
}

TYPED_TEST(HausdorffTest, TooFewPoints)
{
    using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({ 0 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 0 });
    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 1 });

    EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, space_offsets), cuspatial::logic_error);
}

TYPED_TEST(HausdorffTest, FromPython)
{
  using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({ 0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 1.0, 2.0, 3.0, 5.0, 7.0, 0.0, 2.0, 3.0, 6.0, 1.0, 3.0, 6.0 });
    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 5, 9 });

    auto expected = cudf::test::fixed_width_column_wrapper<T>({
        0.0000000000000000, 4.1231056256176606, 4.0000000000000000,
        3.6055512754639896, 0.0000000000000000, 1.4142135623730951,
        4.4721359549995796, 1.4142135623730951, 0.0000000000000000,
    });

    auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

    // auto non_result = thrust::make_constant_iterator<T>(-1);
    // auto expected = cudf::test::fixed_width_column_wrapper<T>(non_result, non_result + actual->size());

    expect_columns_equivalent(expected, actual->view(), true);
}
