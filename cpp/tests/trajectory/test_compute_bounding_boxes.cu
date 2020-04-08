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

  auto bounding_boxes =
      cuspatial::experimental::compute_bounding_boxes(xs, ys, lengths, offsets, this->mr());

  auto h_xs = cudf::test::to_host<double>(xs).first;
  auto h_ys = cudf::test::to_host<double>(ys).first;
  auto h_ts = cudf::test::to_host<cudf::timestamp_ms>(ts).first;
  auto h_id = cudf::test::to_host<int32_t>(grouped->get_column(0)).first;
  auto h_lengths = cudf::test::to_host<int32_t>(lengths).first;
  auto h_offsets = cudf::test::to_host<int32_t>(offsets).first;

  std::vector<double> bbox_x1(h_id.size());
  std::vector<double> bbox_y1(h_id.size());
  std::vector<double> bbox_x2(h_id.size());
  std::vector<double> bbox_y2(h_id.size());

  // compute expected bounding boxes
  for (auto id : h_id) {
    int32_t len = h_lengths[id];
    int32_t idx = h_offsets[id];
    int32_t end = len + idx - 1;
    auto x1 = h_xs[idx];
    auto y1 = h_ys[idx];
    auto x2 = h_xs[idx];
    auto y2 = h_ys[idx];

    for (int32_t i = idx; ++i < end;) {
      x1 = std::min(x1, h_xs[i]);
      y1 = std::min(y1, h_ys[i]);
      x2 = std::max(x2, h_xs[i]);
      y2 = std::max(y2, h_ys[i]);
    }

    bbox_x1[id] = x1;
    bbox_y1[id] = y1;
    bbox_x2[id] = x2;
    bbox_y2[id] = y2;
  }

  cudf::test::expect_columns_equivalent(
      bounding_boxes->get_column(0),
      cudf::test::fixed_width_column_wrapper<double>(bbox_x1.begin(),
                                                     bbox_x1.end()));
  cudf::test::expect_columns_equivalent(
      bounding_boxes->get_column(1),
      cudf::test::fixed_width_column_wrapper<double>(bbox_y1.begin(),
                                                     bbox_y1.end()));
  cudf::test::expect_columns_equivalent(
      bounding_boxes->get_column(2),
      cudf::test::fixed_width_column_wrapper<double>(bbox_x2.begin(),
                                                     bbox_x2.end()));
  cudf::test::expect_columns_equivalent(
      bounding_boxes->get_column(3),
      cudf::test::fixed_width_column_wrapper<double>(bbox_y2.begin(),
                                                     bbox_y2.end()));
}
