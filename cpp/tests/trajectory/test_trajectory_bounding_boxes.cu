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

#include <cudf/utilities/traits.hpp>

#include <cudf/utilities/test/column_utilities.hpp>
#include <cudf/utilities/test/type_lists.hpp>

#include "tests/trajectory/trajectory_utilities.cuh"

template <typename T>
struct TrajectoryBoundingBoxesTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(TrajectoryBoundingBoxesTest, cudf::test::FloatingPointTypes);

constexpr cudf::size_type size{1000};

TYPED_TEST(TrajectoryBoundingBoxesTest, ComputeBoundingBoxesForThreeTrajectories)
{
  using T = TypeParam;

  auto test_data = cuspatial::test::make_test_trajectories_table<T>(size, this->mr());

  std::unique_ptr<cudf::column> offsets;
  std::unique_ptr<cudf::table> sorted;

  std::tie(sorted, offsets) = cuspatial::derive_trajectories(test_data->get_column(0),
                                                             test_data->get_column(1),
                                                             test_data->get_column(2),
                                                             test_data->get_column(3),
                                                             this->mr());

  auto id = sorted->get_column(0);
  auto xs = sorted->get_column(1);
  auto ys = sorted->get_column(2);

  auto bounding_boxes =
    cuspatial::trajectory_bounding_boxes(offsets->size(), id, xs, ys, this->mr());

  auto h_xs      = cudf::test::to_host<T>(xs).first;
  auto h_ys      = cudf::test::to_host<T>(ys).first;
  auto h_offsets = cudf::test::to_host<int32_t>(*offsets).first;

  std::vector<T> bbox_x1(h_offsets.size());
  std::vector<T> bbox_y1(h_offsets.size());
  std::vector<T> bbox_x2(h_offsets.size());
  std::vector<T> bbox_y2(h_offsets.size());

  // compute expected bounding boxes
  for (size_t tid = 0; tid < h_offsets.size(); ++tid) {
    auto idx = h_offsets[tid];
    auto end = tid == h_offsets.size() - 1 ? h_xs.size() : h_offsets[tid + 1];

    auto x1 = h_xs[idx];
    auto y1 = h_ys[idx];
    auto x2 = h_xs[idx];
    auto y2 = h_ys[idx];

    for (size_t i = idx; ++i < end;) {
      x1 = std::min(x1, h_xs[i]);
      y1 = std::min(y1, h_ys[i]);
      x2 = std::max(x2, h_xs[i]);
      y2 = std::max(y2, h_ys[i]);
    }

    bbox_x1[tid] = x1;
    bbox_y1[tid] = y1;
    bbox_x2[tid] = x2;
    bbox_y2[tid] = y2;
  }

  auto x1_actual = bounding_boxes->get_column(0);
  auto y1_actual = bounding_boxes->get_column(1);
  auto x2_actual = bounding_boxes->get_column(2);
  auto y2_actual = bounding_boxes->get_column(3);
  cudf::test::fixed_width_column_wrapper<T> x1_expected(bbox_x1.begin(), bbox_x1.end());
  cudf::test::fixed_width_column_wrapper<T> y1_expected(bbox_y1.begin(), bbox_y1.end());
  cudf::test::fixed_width_column_wrapper<T> x2_expected(bbox_x2.begin(), bbox_x2.end());
  cudf::test::fixed_width_column_wrapper<T> y2_expected(bbox_y2.begin(), bbox_y2.end());

  cudf::test::expect_columns_equivalent(x1_actual, x1_expected);
  cudf::test::expect_columns_equivalent(y1_actual, y1_expected);
  cudf::test::expect_columns_equivalent(x2_actual, x2_expected);
  cudf::test::expect_columns_equivalent(y2_actual, y2_expected);
}
