/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <random>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <vector>

#include <cuspatial/legacy/trajectory.hpp>

struct TrajectoryDerive : public GdfTest {
};

template <typename T>
using wrapper = cudf::test::column_wrapper<T>;

constexpr gdf_size_type column_size{1000};

TEST_F(TrajectoryDerive, DeriveThree)
{
  std::vector<int32_t> sequence(column_size);
  std::iota(sequence.begin(), sequence.end(), 0);

  // three sorted trajectories: one with 2/3 of the points, two with 1/6
  std::vector<int32_t> id_vector(column_size);
  std::transform(sequence.cbegin(), sequence.cend(), id_vector.begin(), [](int32_t i) {
    return (i < 2 * column_size / 3) ? 0 : (i < 5 * column_size / 6) ? 1 : 2;
  });

  // timestamp milliseconds
  std::vector<int64_t> ms_vector(sequence.begin(), sequence.end());

  // randomize sequence
  std::seed_seq seed{0};
  std::mt19937 g(seed);

  std::shuffle(sequence.begin(), sequence.end(), g);

  wrapper<double> in_x(column_size,
                       [&](cudf::size_type i) { return static_cast<double>(sequence[i]); });
  wrapper<double> in_y(column_size,
                       [&](cudf::size_type i) { return static_cast<double>(sequence[i]); });
  wrapper<int32_t> in_id(column_size, [&](cudf::size_type i) { return id_vector[sequence[i]]; });
  wrapper<cudf::timestamp> in_ts(column_size, [&](cudf::size_type i) {
    return static_cast<cudf::timestamp>(ms_vector[sequence[i]]);
  });

  gdf_column traj_id{}, traj_len{}, traj_offset{};

  gdf_size_type num_traj{0};
  EXPECT_NO_THROW(num_traj = cuspatial::derive_trajectories(
                    in_x, in_y, in_id, in_ts, traj_id, traj_len, traj_offset););

  wrapper<gdf_size_type> expected_traj_id{0, 1, 2};
  // need to round up
  wrapper<gdf_size_type> expected_traj_len{
    2 * column_size / 3, (column_size + 5) / 6, (column_size + 5) / 6};
  // for some reason offset is the inclusive scan (offset[i] is the end of
  // trajectory[i] rather than the beginning)
  wrapper<gdf_size_type> expected_traj_offset{
    2 * column_size / 3, 5 * column_size / 6, column_size};

  EXPECT_EQ(num_traj, 3);
  EXPECT_TRUE(expected_traj_id == traj_id);
  EXPECT_TRUE(expected_traj_len == traj_len);
  EXPECT_TRUE(expected_traj_offset == traj_offset);
}

TEST_F(TrajectoryDerive, BadData)
{
  gdf_column out_id, out_len, out_offset;

  gdf_column bad_x, bad_y, bad_in_id, bad_timestamp;
  gdf_column_view(&bad_x, 0, 0, 0, GDF_FLOAT64);
  gdf_column_view(&bad_y, 0, 0, 0, GDF_FLOAT64);
  gdf_column_view(&bad_in_id, 0, 0, 0, GDF_INT32);
  gdf_column_view(&bad_timestamp, 0, 0, 0, GDF_TIMESTAMP);

  // null pointers
  CUDF_EXPECT_THROW_MESSAGE(cuspatial::derive_trajectories(
                              bad_x, bad_y, bad_in_id, bad_timestamp, out_id, out_len, out_offset),
                            "Null input data");

  // size mismatch
  bad_x.data = bad_y.data = bad_in_id.data = bad_timestamp.data =
    reinterpret_cast<void*>(0x0badf00d);
  bad_x.size         = 10;
  bad_y.size         = 12;  // mismatch
  bad_in_id.size     = 10;
  bad_timestamp.size = 10;

  CUDF_EXPECT_THROW_MESSAGE(cuspatial::derive_trajectories(
                              bad_x, bad_y, bad_in_id, bad_timestamp, out_id, out_len, out_offset),
                            "Data size mismatch");

  // Invalid ID datatype
  bad_y.size      = 10;
  bad_in_id.dtype = GDF_FLOAT32;

  CUDF_EXPECT_THROW_MESSAGE(cuspatial::derive_trajectories(
                              bad_x, bad_y, bad_in_id, bad_timestamp, out_id, out_len, out_offset),
                            "Invalid trajectory ID datatype");

  bad_in_id.dtype     = GDF_INT32;
  bad_timestamp.dtype = GDF_DATE32;

  CUDF_EXPECT_THROW_MESSAGE(cuspatial::derive_trajectories(
                              bad_x, bad_y, bad_in_id, bad_timestamp, out_id, out_len, out_offset),
                            "Invalid timestamp datatype");

  bad_timestamp.dtype = GDF_TIMESTAMP;
  bad_x.null_count    = 5;
  CUDF_EXPECT_THROW_MESSAGE(cuspatial::derive_trajectories(
                              bad_x, bad_y, bad_in_id, bad_timestamp, out_id, out_len, out_offset),
                            "NULL support unimplemented");
}
