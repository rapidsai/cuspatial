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
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cuspatial/hausdorff.hpp>
#include <cuspatial/error.hpp>

#include <thrust/iterator/constant_iterator.h>

using namespace cudf;
using namespace test;

template <typename T>
struct HausdorffTest : public BaseFixture {
};

using TestTypes = Types<double>;

TYPED_TEST_CASE(HausdorffTest, TestTypes);

// TYPED_TEST(HausdorffTest, Empty)
// {
//     using T = TypeParam;

//     auto x = cudf::test::fixed_width_column_wrapper<T>({});
//     auto y = cudf::test::fixed_width_column_wrapper<T>({});
//     auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({});

//     auto expected = cudf::test::fixed_width_column_wrapper<T>({});

//     auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

//     expect_columns_equal(expected, actual->view());
// }

// TYPED_TEST(HausdorffTest, SingleTrajectorySinglePoint)
// {
//     using T = TypeParam;

//     auto x = cudf::test::fixed_width_column_wrapper<T>({  152.2 });
//     auto y = cudf::test::fixed_width_column_wrapper<T>({ 2351.0 });
//     auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0 });

//     auto expected = cudf::test::fixed_width_column_wrapper<T>({ 0 });

//     auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

//     expect_columns_equal(expected, actual->view());
// }

// TYPED_TEST(HausdorffTest, TwoShortSpaces)
// {
//     using T = TypeParam;

//     auto x = cudf::test::fixed_width_column_wrapper<T>({ 0, 5, 4 });
//     auto y = cudf::test::fixed_width_column_wrapper<T>({ 0, 12, 3 });
//     auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 1 });

//     auto expected = cudf::test::fixed_width_column_wrapper<T>({ 0, 5,
//                                                                 13, 0 });

//     auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

//     expect_columns_equal(expected, actual->view(), true);
// }

// TYPED_TEST(HausdorffTest, 10kSpacesSinglePoint)
// {
//     using T = TypeParam;

//     constexpr cudf::size_type num_spaces = 10000;
//     constexpr cudf::size_type elements_per_space = 1;
//     constexpr cudf::size_type num_elements = elements_per_space * num_spaces;

//     auto zero_iter = thrust::make_constant_iterator<T>(0);
//     auto counting_iter = thrust::make_counting_iterator<cudf::size_type>(0);
//     auto space_offset_iter = thrust::make_transform_iterator(
//         counting_iter,
//         [elements_per_space](auto idx){ return idx * elements_per_space; });

//     auto x = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
//     auto y = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
//     auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>(space_offset_iter, space_offset_iter + num_spaces);

//     auto expected = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + (num_spaces * num_spaces));

//     auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

//     expect_columns_equal(expected, actual->view(), true);
// }

// TYPED_TEST(HausdorffTest, 2Spaces500kPoints)
// {
//     using T = TypeParam;

//     constexpr cudf::size_type num_spaces = 2;
//     constexpr cudf::size_type elements_per_space = 500000;
//     constexpr cudf::size_type num_elements = elements_per_space * num_spaces;

//     auto zero_iter = thrust::make_constant_iterator<T>(0);
//     auto counting_iter = thrust::make_counting_iterator<cudf::size_type>(0);
//     auto space_offset_iter = thrust::make_transform_iterator(
//         counting_iter,
//         [elements_per_space](auto idx){ return idx * elements_per_space; });

//     auto x = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
//     auto y = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
//     auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>(space_offset_iter, space_offset_iter + num_spaces);

//     auto expected = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + (num_spaces * num_spaces));

//     auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

//     expect_columns_equal(expected, actual->view(), true);
// }

// TYPED_TEST(HausdorffTest, MoreSpacesThanPoints)
// {
//     using T = TypeParam;

//     auto x = cudf::test::fixed_width_column_wrapper<T>({ 0 });
//     auto y = cudf::test::fixed_width_column_wrapper<T>({ 0 });
//     auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 1 });

//     EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, space_offsets), cuspatial::logic_error);
// }

// TYPED_TEST(HausdorffTest, TooFewPoints)
// {
//     using T = TypeParam;

//     auto x = cudf::test::fixed_width_column_wrapper<T>({ 0 });
//     auto y = cudf::test::fixed_width_column_wrapper<T>({ 0 });
//     auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 1 });

//     EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, space_offsets), cuspatial::logic_error);
// }

TYPED_TEST(HausdorffTest, FromPython)
{
    using T = TypeParam;

    // auto x = cudf::test::fixed_width_column_wrapper<T>({ 0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0 });
    // auto y = cudf::test::fixed_width_column_wrapper<T>({ 1.0, 2.0, 3.0, 5.0, 7.0, 0.0, 2.0, 3.0, 6.0, 1.0, 3.0, 6.0 });
    auto x = cudf::test::fixed_width_column_wrapper<T>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });
    auto y = cudf::test::fixed_width_column_wrapper<T>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });
    auto space_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({ 0, 4, 7 });

    auto expected = cudf::test::fixed_width_column_wrapper<T>({});

    auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets);

    expect_columns_equal(expected, actual->view());
}
