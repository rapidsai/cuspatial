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

#include <time.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <utilities/error_utils.hpp>
#include <cuspatial/types.hpp> 
#include <cuspatial/trajectory.hpp> 
#include <utility/utility.hpp>
#include <utility/trajectory_thrust.cuh>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

struct TrajectorySubsetTest : public GdfTest 
{
};

template <typename T>
using wrapper = cudf::test::column_wrapper<T>;

constexpr gdf_size_type column_size{1000};

void test_subset(std::vector<int32_t> ids_to_keep)
{
    std::vector<int32_t> sequence(column_size);
    std::iota(sequence.begin(), sequence.end(), 0);

    //three sorted trajectories: one with 2/3 of the points, two with 1/6
    std::vector<int32_t> point_id(column_size);
    std::transform(sequence.cbegin(), sequence.cend(), point_id.begin(),
                   [](int32_t i) { return (i < 2 * i / 3) ? 0 : 
                                          (i < 5 * i / 6) ? 1 : 2; });

    int point_ms[] = {1,2,3,4,5,1,2,3,4,1,2,3};

    //randomize sequence
    std::seed_seq seed{0};
    std::mt19937 g(seed);
    
    std::shuffle(sequence.begin(), sequence.end(), g);

    wrapper<double> point_x_wrapper(column_size,
        [&](gdf_index_type i) { return static_cast<double>(sequence[i]); });
    wrapper<double> point_y_wrapper(column_size,
        [&](gdf_index_type i) { return static_cast<double>(sequence[i]); });
    wrapper<int32_t> point_id_wrapper(column_size,
        [&](gdf_index_type i) { return point_id[sequence[i]]; });
    wrapper<cudf::timestamp> point_ts_wrapper(column_size,
        [&](gdf_index_type i) { 
            return static_cast<cudf::timestamp>(point_ms[sequence[i]]); 
        });
    wrapper<int32_t> ids_wrapper{ids_to_keep};

    gdf_column out_x{}, out_y{}, out_id{}, out_ts{};

    // sort the ids to keep now that we've copied them unsorted to input column
    std::sort(ids_to_keep.begin(), ids_to_keep.end());

    std::vector<int32_t> expected_sequence(sequence.size());
    auto expected_size =
        std::copy_if(sequence.begin(), sequence.end(), expected_sequence.begin(),
            [&](int32_t i) {
                return std::binary_search(ids_to_keep.begin(), ids_to_keep.end(),
                                          point_id[i]);
            }
        ) - expected_sequence.begin();

    wrapper<double> expected_x(expected_size,
        [&](gdf_index_type i) { return static_cast<double>(expected_sequence[i]); });
    wrapper<double> expected_y(expected_size,
        [&](gdf_index_type i) { return static_cast<double>(expected_sequence[i]); });
    wrapper<int32_t> expected_id(expected_size,
        [&](gdf_index_type i) { return point_id[expected_sequence[i]]; });
    wrapper<cudf::timestamp> expected_ts(expected_size,
        [&](gdf_index_type i) {
            return static_cast<cudf::timestamp>(point_ms[expected_sequence[i]]);
        });

    gdf_size_type num_hit{0};
    
    EXPECT_NO_THROW(
        num_hit = cuspatial::subset_trajectory_id(ids_wrapper,
                                                  point_x_wrapper,
                                                  point_y_wrapper,
                                                  point_id_wrapper,
                                                  point_ts_wrapper,
                                                  out_x, out_y,
                                                  out_id, out_ts));

    EXPECT_EQ(num_hit, expected_size);
    EXPECT_TRUE(expected_x == out_x);
    EXPECT_TRUE(expected_y == out_y);
    EXPECT_TRUE(expected_id == out_id);
    EXPECT_TRUE(expected_ts == out_ts);
}

TEST_F(TrajectorySubsetTest, SelectSome)
{
    std::vector<int32_t> keep_all{0, 1, 2};
    test_subset(keep_all);
    std::vector<int32_t> keep_all_unsorted{2, 0, 1};
    test_subset(keep_all_unsorted);
    std::vector<int32_t> keep_two{1, 2};
    test_subset(keep_two);
    std::vector<int32_t> keep_one{1};
    test_subset(keep_one);
    std::vector<int32_t> keep_none{};
    test_subset(keep_none);
}

TEST_F(TrajectorySubsetTest, BadData)
{
    //constexpr gdf_size_type column_size{1000};

    gdf_column out_x, out_y, out_id, out_timestamp;

    gdf_column bad_x, bad_y, bad_in_id, bad_timestamp, bad_id;
    gdf_column_view(&bad_id, 0, 0, 0, GDF_INT32);
    gdf_column_view(&bad_x, 0, 0, 0,  GDF_FLOAT64);
    gdf_column_view(&bad_y, 0, 0, 0,  GDF_FLOAT64);
    gdf_column_view(&bad_in_id, 0, 0, 0, GDF_INT32);
    gdf_column_view(&bad_timestamp, 0, 0, 0, GDF_TIMESTAMP);

    // null pointers
    CUDF_EXPECT_THROW_MESSAGE(cuspatial::subset_trajectory_id(bad_id,
                                                              bad_x, bad_y,
                                                              bad_in_id,
                                                              bad_timestamp,
                                                              out_x, out_y, 
                                                              out_id,
                                                              out_timestamp),
        "Null input data");
    
    // size mismatch
    bad_x.data = bad_y.data = bad_in_id.data = bad_timestamp.data = 
        reinterpret_cast<void*>(0x0badf00d);
    bad_x.size = 10;
    bad_y.size = 12; // mismatch
    bad_in_id.size = 10;
    bad_timestamp.size = 10;
    
    CUDF_EXPECT_THROW_MESSAGE(cuspatial::subset_trajectory_id(bad_id,
                                                              bad_x, bad_y,
                                                              bad_in_id,
                                                              bad_timestamp,
                                                              out_x, out_y, 
                                                              out_id,
                                                              out_timestamp),
        "Data size mismatch");

    // Invalid ID datatype
    bad_y.size = 10;
    bad_in_id.dtype = GDF_FLOAT32;

    CUDF_EXPECT_THROW_MESSAGE(cuspatial::subset_trajectory_id(bad_id,
                                                              bad_x, bad_y,
                                                              bad_in_id,
                                                              bad_timestamp,
                                                              out_x, out_y, 
                                                              out_id,
                                                              out_timestamp),
        "Invalid trajectory ID datatype");

    bad_in_id.dtype = GDF_INT32;
    bad_id.dtype = GDF_INT8;

    CUDF_EXPECT_THROW_MESSAGE(cuspatial::subset_trajectory_id(bad_id,
                                                              bad_x, bad_y,
                                                              bad_in_id,
                                                              bad_timestamp,
                                                              out_x, out_y, 
                                                              out_id,
                                                              out_timestamp),
        "Trajectory ID datatype mismatch");

    bad_id.dtype = GDF_INT32;
    bad_timestamp.dtype = GDF_DATE32;

    CUDF_EXPECT_THROW_MESSAGE(cuspatial::subset_trajectory_id(bad_id,
                                                              bad_x, bad_y,
                                                              bad_in_id,
                                                              bad_timestamp,
                                                              out_x, out_y, 
                                                              out_id,
                                                              out_timestamp),
        "Invalid timestamp datatype");

    bad_timestamp.dtype = GDF_TIMESTAMP;
    bad_x.null_count = 5;
    CUDF_EXPECT_THROW_MESSAGE(cuspatial::subset_trajectory_id(bad_id,
                                                              bad_x, bad_y,
                                                              bad_in_id,
                                                              bad_timestamp,
                                                              out_x, out_y, 
                                                              out_id,
                                                              out_timestamp),
        "NULL support unimplemented");
}
