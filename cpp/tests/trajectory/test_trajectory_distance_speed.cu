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

#include <vector>
#include <gtest/gtest.h>

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>

#include <cuspatial/trajectory.hpp> 

struct TrajectoryDistanceSpeed : public GdfTest 
{
};

template <typename T>
using wrapper = cudf::test::column_wrapper<T>;

constexpr gdf_size_type column_size{100};

constexpr int m_per_km = 1000;
constexpr int ms_per_second = 1000;
constexpr int ms_per_hour = ms_per_second * 60 * 60;

TEST_F(TrajectoryDistanceSpeed, DistanceAndSpeedThree)
{
    std::vector<int32_t> sequence(column_size);
    std::iota(sequence.begin(), sequence.end(), 0);

    //three sorted trajectories: one with 2/3 of the points, two with 1/6
    std::vector<int32_t> id_vector(column_size);
    std::transform(sequence.cbegin(), sequence.cend(), id_vector.begin(),
                   [](int32_t i) { 
                       return (i < 2 * column_size / 3) ? 0 : 
                              (i < 5 * column_size / 6) ? 1 : 2; 
                    });

    // timestamp milliseconds
    std::vector<int64_t> ms_vector(column_size);
    std::transform(sequence.cbegin(), sequence.cend(), ms_vector.begin(), 
                   [](int32_t i) { return i * ms_per_hour + i; }); 

    std::vector<double> x(sequence.begin(), sequence.end());
    wrapper<double> in_x{x};
        //[&](gdf_index_type i) { return static_cast<double>(sequence[i]); });
    wrapper<double> in_y(column_size,
        [&](gdf_index_type i) { return static_cast<double>(sequence[i]); });
    wrapper<cudf::timestamp> in_ts(column_size,
        [&](gdf_index_type i) {
            return static_cast<cudf::timestamp>(ms_vector[sequence[i]]);
        });

    std::vector<gdf_size_type> trajectory_length{2 * column_size / 3, 
                                                 (column_size + 5) / 6,
                                                 (column_size + 5) / 6};
    std::vector<gdf_size_type> trajectory_offset{2 * column_size / 3,
                                                 5 * column_size / 6,
                                                 column_size};
    wrapper<gdf_size_type> traj_len(trajectory_length);
    wrapper<gdf_size_type> traj_offset{trajectory_offset};

    std::pair<gdf_column, gdf_column> distance_speed;

    EXPECT_NO_THROW(
     distance_speed = cuspatial::trajectory_distance_and_speed(in_x, in_y,
                                                               in_ts, traj_len,
                                                               traj_offset);
    );

    EXPECT_EQ(distance_speed.first.size, 3);
    EXPECT_EQ(distance_speed.second.size, 3);

    std::vector<gdf_size_type> id{0, 1, 2};
    std::vector<double> distance(3, 0);
    std::vector<double> time(3, 0);
    std::vector<double> speed(3, 0);
    
    // compute expected distance and speed
    for (auto i : id) {
        gdf_size_type length = trajectory_length[i];
        gdf_size_type begin = (i == 0) ? 0 : trajectory_offset[i - 1];
        gdf_size_type end = trajectory_offset[i];
        EXPECT_EQ(end, begin + length);

        for (gdf_size_type j = 0; j < length-1; j++) {
            double d = static_cast<double>(sequence[begin + j + 1] -
                                           sequence[begin + j]);
            double ms = static_cast<double>(ms_vector[begin + j + 1] -
                                            ms_vector[begin + j]);
            d = sqrt(2 * d * d); // since x == y
            distance[i] += d * m_per_km;
            time[i] += (ms / ms_per_second);
        }
        speed[i] = distance[i] / time[i];
    }

    std::vector<double> gpu_distance(distance_speed.first.size);
    cudaMemcpy(gpu_distance.data(), distance_speed.first.data,
               distance_speed.first.size * sizeof(double), cudaMemcpyDefault);
    std::vector<double> gpu_speed(distance_speed.second.size);
    cudaMemcpy(gpu_speed.data(), distance_speed.second.data,
               distance_speed.second.size * sizeof(double), cudaMemcpyDefault);
    for (size_t i = 0; i < gpu_distance.size(); i++) {
        
        EXPECT_NEAR(gpu_distance[i], distance[i], 1e-9);
        EXPECT_NEAR(gpu_speed[i], speed[i], 1e-9);
    }
}

TEST_F(TrajectoryDistanceSpeed, BadData)
{
    gdf_column bad_x, bad_y, bad_timestamp, bad_length, bad_offset;
    gdf_column_view(&bad_x, 0, 0, 0,  GDF_FLOAT64);
    gdf_column_view(&bad_y, 0, 0, 0,  GDF_FLOAT64);
    gdf_column_view(&bad_timestamp, 0, 0, 0, GDF_TIMESTAMP);
    gdf_column_view(&bad_length, 0, 0, 0, GDF_INT32);
    gdf_column_view(&bad_offset, 0, 0, 0, GDF_INT32);

    // null pointers
    CUDF_EXPECT_THROW_MESSAGE(
        cuspatial::trajectory_distance_and_speed(bad_x, bad_y, bad_timestamp, 
                                                 bad_length, bad_offset),
        "Null input data");

    // size mismatch
    bad_x.data = bad_y.data = bad_timestamp.data = bad_length.data =
        bad_offset.data = reinterpret_cast<void*>(0x0badf00d);
    bad_x.size = 10;
    bad_y.size = 12; // mismatch
    bad_timestamp.size = 10;
    bad_length.size = 3;
    bad_offset.size = 3;

    CUDF_EXPECT_THROW_MESSAGE(
        cuspatial::trajectory_distance_and_speed(bad_x, bad_y, bad_timestamp, 
                                                 bad_length, bad_offset),
        "Data size mismatch");

    // Invalid ID datatype
    bad_y.size = 10;
    bad_offset.size = 4;

    CUDF_EXPECT_THROW_MESSAGE(
        cuspatial::trajectory_distance_and_speed(bad_x, bad_y, bad_timestamp, 
                                                 bad_length, bad_offset),
        "Data size mismatch");

    bad_offset.size = 3;
    bad_length.dtype = GDF_FLOAT32;

    CUDF_EXPECT_THROW_MESSAGE(
        cuspatial::trajectory_distance_and_speed(bad_x, bad_y, bad_timestamp, 
                                                 bad_length, bad_offset),
        "Invalid trajectory length datatype");
    
    bad_length.dtype = GDF_INT32;
    bad_offset.dtype = GDF_FLOAT32;

    CUDF_EXPECT_THROW_MESSAGE(
        cuspatial::trajectory_distance_and_speed(bad_x, bad_y, bad_timestamp, 
                                                 bad_length, bad_offset),
        "Invalid trajectory offset datatype");

    bad_offset.dtype = GDF_INT32;
    bad_timestamp.dtype = GDF_DATE32;

    CUDF_EXPECT_THROW_MESSAGE(
        cuspatial::trajectory_distance_and_speed(bad_x, bad_y, bad_timestamp, 
                                                 bad_length, bad_offset),
        "Invalid timestamp datatype");

    bad_timestamp.dtype = GDF_TIMESTAMP;
    bad_x.null_count = 5;
    CUDF_EXPECT_THROW_MESSAGE(
        cuspatial::trajectory_distance_and_speed(bad_x, bad_y, bad_timestamp, 
                                                 bad_length, bad_offset),
        "NULL support unimplemented");

    bad_x.null_count = 0;
    bad_x.size = 1;
    bad_y.size = 1;
    bad_timestamp.size = 1;
    CUDF_EXPECT_THROW_MESSAGE(
        cuspatial::trajectory_distance_and_speed(bad_x, bad_y, bad_timestamp, 
                                                 bad_length, bad_offset),
        "Insufficient trajectory data");

    
}