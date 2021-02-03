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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include "trajectory_utilities.cuh"

struct DeriveTrajectoriesTest : public cudf::test::BaseFixture {
};

constexpr cudf::size_type size{1000};

TEST_F(DeriveTrajectoriesTest, DerivesThreeTrajectories)
{
  auto sorted  = cuspatial::test::make_test_trajectories_table<double>(size, this->mr());
  auto id      = sorted->get_column(0);
  auto xs      = sorted->get_column(1);
  auto ys      = sorted->get_column(2);
  auto ts      = sorted->get_column(3);
  auto results = cuspatial::derive_trajectories(id, xs, ys, ts, this->mr());
  cudf::test::expect_tables_equal(*results.first, *sorted);
  cudf::test::expect_columns_equal(
    *results.second,
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 2 * size / 3, 5 * size / 6});
}
