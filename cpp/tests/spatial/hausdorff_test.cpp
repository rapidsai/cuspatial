/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cuspatial/distance/hausdorff.hpp>
#include <cuspatial/error.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <vector>

using namespace cudf;
using namespace test;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

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

TYPED_TEST(HausdorffTest, MoreSpacesThanPoints)
{
  using T = TypeParam;

  auto x             = cudf::test::fixed_width_column_wrapper<T>({0});
  auto y             = cudf::test::fixed_width_column_wrapper<T>({0});
  auto space_offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 1});

  EXPECT_THROW(cuspatial::directed_hausdorff_distance(x, y, space_offsets, this->mr()),
               cuspatial::logic_error);
}
