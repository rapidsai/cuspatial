/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuspatial/error.hpp>
#include <cuspatial/projection.hpp>
#include <cuspatial/spatial/allpairs_multipoint_equals_count.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <type_traits>

using namespace cudf::test;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
struct AllpairsMultipointEqualsCountTest : public BaseFixture {
};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = Types<double>;
TYPED_TEST_CASE(AllpairsMultipointEqualsCountTest, TestTypes);

TYPED_TEST(AllpairsMultipointEqualsCountTest, Single)
{
  using T  = TypeParam;
  auto lhs = fixed_width_column_wrapper<T>({});
  auto rhs = fixed_width_column_wrapper<T>({});

  auto output = cuspatial::allpairs_multipoint_equals_count(lhs, rhs);

  auto expected = fixed_width_column_wrapper<T>({});

  expect_columns_equivalent(expected, output->view(), verbosity);
}
