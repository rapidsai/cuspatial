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

#include "tests/trajectory/trajectory_utilities.cuh"

struct DeriveTrajectoriesTest : public cudf::test::BaseFixture {};

constexpr cudf::size_type size{1000};

TEST_F(DeriveTrajectoriesTest, DerivesThreeTrajectories) {
  auto sorted = cuspatial::test::make_test_trajectories_table(size);
  auto result = cuspatial::experimental::derive_trajectories(
      sorted->get_column(0), this->mr());

  auto ids = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2};
  auto lengths = cudf::test::fixed_width_column_wrapper<int32_t>{
      2 * size / 3, (size + 5) / 6, (size + 5) / 6};
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>{
      0, 2 * size / 3, 5 * size / 6};
  
  cudf::test::expect_columns_equal(result->get_column(0), ids);
  cudf::test::expect_columns_equal(result->get_column(1), lengths);
  cudf::test::expect_columns_equal(result->get_column(2), offsets);
}
