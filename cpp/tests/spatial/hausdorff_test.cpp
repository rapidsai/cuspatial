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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cuspatial/hausdorff.hpp>

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

    auto x = cudf::test::fixed_width_column_wrapper<T>({});
    auto y = cudf::test::fixed_width_column_wrapper<T>({});
    auto trajectories = cudf::test::fixed_width_column_wrapper<cudf::size_type>({});

    auto expected = cudf::test::fixed_width_column_wrapper<T>({});

    auto actual = cuspatial::directed_hausdorff_distance(x, y, trajectories);

    expect_columns_equal(expected, actual->view());
}

TYPED_TEST(HausdorffTest, SingleElementTrajectory)
{
    using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({  152.2 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 2351.0 });
    auto trajectories = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 1 });

    auto expected = cudf::test::fixed_width_column_wrapper<T>({ 0 });

    auto actual = cuspatial::directed_hausdorff_distance(x, y, trajectories);

    expect_columns_equal(expected, actual->view());
}

TYPED_TEST(HausdorffTest, Old)
{
    using T = TypeParam;

    auto x = cudf::test::fixed_width_column_wrapper<T>({ 0, 5, 4 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 0, 12, 3 });
    auto trajectories = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 1, 2 });

    auto expected = cudf::test::fixed_width_column_wrapper<T>({ 0, 5,
                                                                13, 0 });

    auto actual = cuspatial::directed_hausdorff_distance(x, y, trajectories);

    expect_columns_equal(expected, actual->view(), true);
}
