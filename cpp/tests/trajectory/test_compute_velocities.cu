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

  auto grouped = cuspatial::experimental::derive_trajectories(id, this->mr());
  auto lengths = grouped->get_column(1);
  auto offsets = grouped->get_column(2);

  auto velocity =
      cuspatial::experimental::compute_velocities(xs, ys, ts, lengths, offsets);

  auto h_xs = cudf::test::to_host<double>(xs).first;
  auto h_ys = cudf::test::to_host<double>(ys).first;
  auto h_ts = cudf::test::to_host<cudf::timestamp_ms>(ts).first;
  auto h_id = cudf::test::to_host<int32_t>(grouped->get_column(0)).first;
  auto h_lengths = cudf::test::to_host<int32_t>(lengths).first;
  auto h_offsets = cudf::test::to_host<int32_t>(offsets).first;

  std::vector<double> dist(h_id.size());
  std::vector<double> speed(h_id.size());

  // compute expected distance and speed
  for (auto id : h_id) {
    cudf::size_type len = h_lengths[id];
    cudf::size_type idx = h_offsets[id];
    cudf::size_type end = len + idx - 1;
    cudf::timestamp_ms::duration dt{0};
    for (cudf::size_type i = idx; i < end; i++) {
      auto const x0 = h_xs[i + 0];
      auto const x1 = h_xs[i + 1];
      auto const y0 = h_ys[i + 0];
      auto const y1 = h_ys[i + 1];
      dt += (h_ts[i + 1] - h_ts[i]);
      dist[id] += sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2)) * 1000;  // km to m
    }
    speed[id] = dist[id] / (static_cast<double>(dt.count()) / 1000);  // m/s
  }

  cudf::test::expect_columns_equivalent(
      velocity->get_column(0),
      cudf::test::fixed_width_column_wrapper<double>(dist.begin(), dist.end()));
  cudf::test::expect_columns_equivalent(
      velocity->get_column(1), cudf::test::fixed_width_column_wrapper<double>(
                                   speed.begin(), speed.end()));
}
