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
  auto object_id = sorted->get_column(0);
  cudf::test::expect_columns_equal(
      *cuspatial::experimental::compute_trajectory_offsets(object_id, this->mr()),
      cudf::test::fixed_width_column_wrapper<int32_t>{2 * size / 3,
                                                      5 * size / 6, size});
}
