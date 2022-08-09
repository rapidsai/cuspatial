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

#include "cuspatial/error.hpp"
#include "trajectory_utilities.cuh"

#include <cudf/types.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

struct DeriveTrajectoriesTest : public cudf::test::BaseFixture {
};

TEST_F(DeriveTrajectoriesTest, SizeMismatch)
{
  auto sorted = cuspatial::test::make_test_trajectories_table<double>(1000, this->mr());

  {
    auto id = sorted->get_column(0);
    auto xs = sorted->get_column(1);
    // only half the data
    auto ys = cudf::column(
      cudf::device_span<double const>(sorted->get_column(2).view().data<double>(), 500));
    auto ts = sorted->get_column(3);
    EXPECT_THROW(cuspatial::derive_trajectories(id, xs, ys, ts, this->mr()),
                 cuspatial::logic_error);
  }
  {
    auto id = cudf::column(cudf::device_span<cudf::size_type const>(
      sorted->get_column(0).view().data<cudf::size_type>(), 500));
    auto xs = sorted->get_column(1);
    auto ys = sorted->get_column(2);

    auto ts = sorted->get_column(3);
    EXPECT_THROW(cuspatial::derive_trajectories(id, xs, ys, ts, this->mr()),
                 cuspatial::logic_error);
  }
  {
    auto id = sorted->get_column(0);
    auto xs = sorted->get_column(1);
    auto ys = sorted->get_column(2);

    auto ts = cudf::column(cudf::device_span<cudf::timestamp_ms const>(
      sorted->get_column(0).view().data<cudf::timestamp_ms>(), 500));
    EXPECT_THROW(cuspatial::derive_trajectories(id, xs, ys, ts, this->mr()),
                 cuspatial::logic_error);
  }
}

TEST_F(DeriveTrajectoriesTest, TypeError)
{
  auto sorted = cuspatial::test::make_test_trajectories_table<double>(1000, this->mr());

  {
    auto id = sorted->get_column(1);  // not integer
    auto xs = sorted->get_column(1);
    auto ys = sorted->get_column(2);
    auto ts = sorted->get_column(3);
    EXPECT_THROW(cuspatial::derive_trajectories(id, xs, ys, ts, this->mr()),
                 cuspatial::logic_error);
  }

  {
    auto id = sorted->get_column(0);
    auto xs = sorted->get_column(1);
    auto ys = sorted->get_column(2);
    auto ts = sorted->get_column(1);  // not timestamp
    EXPECT_THROW(cuspatial::derive_trajectories(id, xs, ys, ts, this->mr()),
                 cuspatial::logic_error);
  }
}

TEST_F(DeriveTrajectoriesTest, Nulls)
{
  auto sorted = cuspatial::test::make_test_trajectories_table<double>(1000, this->mr());

  {
    auto id    = sorted->get_column(0);
    auto nulls = rmm::device_uvector<int>(1000, rmm::cuda_stream_default);
    cudaMemsetAsync(nulls.data(), 0xcccc, nulls.size(), rmm::cuda_stream_default.value());
    auto nulls_buffer = nulls.release();
    id.set_null_mask(nulls_buffer);
    auto xs = sorted->get_column(1);
    auto ys = sorted->get_column(2);
    auto ts = sorted->get_column(3);
    EXPECT_THROW(cuspatial::derive_trajectories(id, xs, ys, ts, this->mr()),
                 cuspatial::logic_error);
  }
}

TEST_F(DeriveTrajectoriesTest, DerivesThreeTrajectories)
{
  auto sorted  = cuspatial::test::make_test_trajectories_table<double>(1000, this->mr());
  auto id      = sorted->get_column(0);
  auto xs      = sorted->get_column(1);
  auto ys      = sorted->get_column(2);
  auto ts      = sorted->get_column(3);
  auto results = cuspatial::derive_trajectories(id, xs, ys, ts, this->mr());
  cudf::test::expect_tables_equal(*results.first, *sorted);
  cudf::test::expect_columns_equal(
    *results.second,
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 2 * id.size() / 3, 5 * id.size() / 6});
}
