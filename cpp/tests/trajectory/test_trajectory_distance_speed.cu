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

MATCHER_P(FloatNearPointwise, tol, "Out of range") {
    return (std::get<0>(arg) > std::get<1>(arg) - tol &&
            std::get<0>(arg) < std::get<1>(arg) + tol) ;
}

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
    EXPECT_THAT(gpu_distance, testing::Pointwise(FloatNearPointwise(1e-8),
                distance));
    EXPECT_THAT(gpu_speed, testing::Pointwise(FloatNearPointwise(1e-8),
                speed));
}
