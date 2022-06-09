/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/hausdorff.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <vector>

using namespace cudf;
using namespace test;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T, uint32_t num_spaces, uint32_t elements_per_space>
void generic_hausdorff_test(rmm::mr::device_memory_resource* mr)
{
  constexpr uint64_t num_elements =
    static_cast<uint64_t>(elements_per_space) * static_cast<uint64_t>(num_spaces);

  auto zero_iter         = thrust::make_constant_iterator<T>(0);
  auto counting_iter     = thrust::make_counting_iterator<uint32_t>(0);
  auto space_offset_iter = thrust::make_transform_iterator(
    counting_iter, [](auto idx) { return idx * elements_per_space; });

  auto x = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
  auto y = cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + num_elements);
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>(
    space_offset_iter, space_offset_iter + num_spaces);

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>(zero_iter, zero_iter + (num_spaces * num_spaces));

  auto actual = std::move(cuspatial::directed_hausdorff_distance(x, y, space_offsets, mr));

  expect_columns_equivalent(expected, actual->view());
}

template <typename T>
struct HausdorffTest : public BaseFixture {
};

TYPED_TEST_CASE(HausdorffTest, cudf::test::FloatingPointTypes);

TYPED_TEST(HausdorffTest, Empty)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets, this->mr());

  expect_columns_equivalent(expected, actual->view(), verbosity);
}

TYPED_TEST(HausdorffTest, Simple)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({0, 1, 0, 0});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({0, 0, 1, 2});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 2});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0.0, std::sqrt(2.0), 2.0, 0.0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets, this->mr());

  expect_columns_equivalent(expected, actual->view(), verbosity);
}

TYPED_TEST(HausdorffTest, SingleTrajectorySinglePoint)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({152.2});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({2351.0});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({0});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets, this->mr());

  expect_columns_equivalent(expected, actual->view(), verbosity);
}

TYPED_TEST(HausdorffTest, TwoShortSpaces)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({0, 5, 4});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({0, 12, 3});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 1});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0, 5, 13, 0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets, this->mr());

  expect_columns_equivalent(expected, actual->view(), verbosity);
}

TYPED_TEST(HausdorffTest, TwoShortSpaces2)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({1, 5, 4, 2, 3, 7});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({1, 12, 3, 8, 4, 7});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 3, 4});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({0.0,
                                                             7.0710678118654755,
                                                             5.3851648071345037,
                                                             5.0000000000000000,
                                                             0.0,
                                                             4.1231056256176606,
                                                             5.0,
                                                             5.0990195135927854,
                                                             0.0});

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets, this->mr());

  expect_columns_equivalent(expected, actual->view(), verbosity);
}

TYPED_TEST(HausdorffTest, 500Spaces100Points)
{
  generic_hausdorff_test<TypeParam, 500, 100>(this->mr());
}

TYPED_TEST(HausdorffTest, 10000Spaces10Points)
{
  generic_hausdorff_test<TypeParam, 10000, 10>(this->mr());
}

TYPED_TEST(HausdorffTest, 10Spaces10000Points)
{
  generic_hausdorff_test<TypeParam, 10, 10000>(this->mr());
}

TYPED_TEST(HausdorffTest, MoreSpacesThanPoints)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({0});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({0});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 1});

  EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, space_offsets, this->mr()),
               cuspatial::logic_error);
}

TYPED_TEST(HausdorffTest, ThreeSpacesLengths543)
{
  using T = TypeParam;

  auto x = cudf::test::fixed_width_column_wrapper<T>(
    {0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0});
  auto y = cudf::test::fixed_width_column_wrapper<T>(
    {1.0, 2.0, 3.0, 5.0, 7.0, 0.0, 2.0, 3.0, 6.0, 1.0, 3.0, 6.0});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 5, 9});

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

  auto actual = cuspatial::directed_hausdorff_distance(x, y, space_offsets, this->mr());

  expect_columns_equivalent(expected, actual->view(), verbosity);
}
