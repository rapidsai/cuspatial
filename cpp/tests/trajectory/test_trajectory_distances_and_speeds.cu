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

#include <cmath>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include "tests/trajectory/trajectory_utilities.cuh"

template <typename T>
struct TrajectoryDistanceSpeedTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(TrajectoryDistanceSpeedTest, cudf::test::FloatingPointTypes);

constexpr cudf::size_type size{1000};

TYPED_TEST(TrajectoryDistanceSpeedTest,
           ComputeDistanceAndSpeedForThreeTrajectories) {
  using T = TypeParam;

  auto test_data =
      cuspatial::test::make_test_trajectories_table<T>(size, this->mr());

  std::unique_ptr<cudf::column> offsets;
  std::unique_ptr<cudf::experimental::table> sorted;

  std::tie(sorted, offsets) = cuspatial::experimental::derive_trajectories(
      test_data->get_column(0), test_data->get_column(1),
      test_data->get_column(2), test_data->get_column(3), this->mr());

  auto id = sorted->get_column(0);
  auto xs = sorted->get_column(1);
  auto ys = sorted->get_column(2);
  auto ts = sorted->get_column(3);

  auto distance_and_speed =
      cuspatial::experimental::trajectory_distances_and_speeds(
          offsets->size(), id, xs, ys, ts, this->mr());

  using Rep = typename cudf::timestamp_ms::rep;

  auto h_xs = cudf::test::to_host<T>(xs).first;
  auto h_ys = cudf::test::to_host<T>(ys).first;
  auto h_ts = cudf::test::to_host<Rep>(ts).first;
  auto h_offsets = cudf::test::to_host<int32_t>(*offsets).first;

  std::vector<double> speed(h_offsets.size());
  std::vector<double> distance(h_offsets.size());

  // compute expected distance and speed
  for (size_t tid = 0; tid < h_offsets.size(); ++tid) {
    auto idx = h_offsets[tid];
    auto end = tid == h_offsets.size() - 1 ? h_xs.size() : h_offsets[tid + 1];
    Rep time_ms = (h_ts[end - 1] - h_ts[idx]);
    if ((end - idx) < 2 || time_ms == 0) {
      distance[tid] = 0.0;
      speed[tid] = 0.0;
    } else {
      double dist_km{0.0};
      for (size_t i = idx; i < end - 1; i++) {
        auto x0 = static_cast<double>(h_xs[i + 0]);
        auto x1 = static_cast<double>(h_xs[i + 1]);
        auto y0 = static_cast<double>(h_ys[i + 0]);
        auto y1 = static_cast<double>(h_ys[i + 1]);
        dist_km += std::hypot(x1 - x0, y1 - y0);
      }
      distance[tid] = dist_km * 1000.0;                 // km to m
      speed[tid] = (distance[tid] * 1000.0) / time_ms;  // m/s
    }
  }

  auto speed_actual = distance_and_speed->get_column(1);
  auto distance_actual = distance_and_speed->get_column(0);

  auto speed_expected = cudf::test::fixed_width_column_wrapper<double>(
      speed.begin(), speed.end());
  auto distance_expected = cudf::test::fixed_width_column_wrapper<double>(
      distance.begin(), distance.end());

  cudf::test::expect_columns_equivalent(distance_actual, distance_expected);
  cudf::test::expect_columns_equivalent(speed_actual, speed_expected);
}
