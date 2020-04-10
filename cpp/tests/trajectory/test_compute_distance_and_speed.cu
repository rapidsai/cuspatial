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

#include <tests/utilities/column_utilities.hpp>

#include "tests/trajectory/trajectory_utilities.cuh"

struct TrajectoryDistanceSpeedTest : public cudf::test::BaseFixture {};

constexpr cudf::size_type size{1000};

TEST_F(TrajectoryDistanceSpeedTest,
       ComputeDistanceAndSpeedForThreeTrajectories) {
  auto sorted = cuspatial::test::make_test_trajectories_table(size);
  auto id = sorted->get_column(0);
  auto ts = sorted->get_column(1);
  auto xs = sorted->get_column(2);
  auto ys = sorted->get_column(3);

  auto offsets = cuspatial::experimental::compute_trajectory_offsets(id, this->mr());

  auto velocity = cuspatial::experimental::compute_distance_and_speed(
      xs, ys, ts, *offsets, this->mr());

  auto h_xs = cudf::test::to_host<double>(xs).first;
  auto h_ys = cudf::test::to_host<double>(ys).first;
  auto h_ts = cudf::test::to_host<cudf::timestamp_ms>(ts).first;
  auto h_offsets = cudf::test::to_host<int32_t>(*offsets).first;

  std::vector<double> distance(h_offsets.size());
  std::vector<double> speed(h_offsets.size());

  // compute expected distance and speed
  for (size_t tid = 0; tid < h_offsets.size(); ++tid) {
    auto end = h_offsets[tid] - 1;
    auto idx = tid == 0 ? 0 : h_offsets[tid - 1];
    auto time_ms = h_ts[end] - h_ts[idx];
    if ((end - idx) < 2) {
      distance[tid] = -2.0;
      speed[tid] = -2.0;
    } else if (time_ms.count() == 0) {
      distance[tid] = -3.0;
      speed[tid] = -3.0;
    } else {
      double dist_km{0.0};
      for (int32_t i = idx; i < end; i++) {
        auto const x0 = h_xs[i + 0];
        auto const x1 = h_xs[i + 1];
        auto const y0 = h_ys[i + 0];
        auto const y1 = h_ys[i + 1];
        dist_km += sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2));
      }
      distance[tid] = dist_km * 1000;                        // km to m
      speed[tid] = dist_km * 1000000 / time_ms.count();  // m/s
    }
  }

  auto expected_distance =
      cudf::test::fixed_width_column_wrapper<double>(distance.begin(), distance.end());
  auto expected_speed = cudf::test::fixed_width_column_wrapper<double>(
      speed.begin(), speed.end());

  cudf::test::expect_columns_equivalent(velocity->get_column(0),
                                        expected_distance);
  cudf::test::expect_columns_equivalent(velocity->get_column(1),
                                        expected_speed);
}
