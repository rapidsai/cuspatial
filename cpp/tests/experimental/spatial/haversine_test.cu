/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/haversine.hpp>

#include <rmm/device_vector.hpp>

#include <gtest/gtest.h>

template <typename T>
struct HaversineTest : public ::testing::Test {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(HaversineTest, TestTypes);

TYPED_TEST(HaversineTest, Empty)
{
  using T = TypeParam;

  auto a_lon = rmm::device_vector<T>{};
  auto a_lat = rmm::device_vector<T>{};
  auto b_lon = rmm::device_vector<T>{};
  auto b_lat = rmm::device_vector<T>{};

  auto distance = rmm::device_vector<T>{};
  auto expected = rmm::device_vector<T>{};

  auto distance_end = cuspatial::haversine_distance(
    a_lon.begin(), a_lon.end(), a_lat.begin(), b_lon.begin(), b_lat.begin(), distance.begin());

  EXPECT_EQ(distance, expected);
  EXPECT_EQ(0, std::distance(distance.begin(), distance_end));
}

TYPED_TEST(HaversineTest, Zero)
{
  using T = TypeParam;

  auto a_lon = rmm::device_vector<T>{1, 0};
  auto a_lat = rmm::device_vector<T>{1, 0};
  auto b_lon = rmm::device_vector<T>{1, 0};
  auto b_lat = rmm::device_vector<T>{1, 0};

  auto distance = rmm::device_vector<T>{1, -1};
  auto expected = rmm::device_vector<T>{1, 0};

  auto distance_end = cuspatial::haversine_distance(
    a_lon.begin(), a_lon.end(), a_lat.begin(), b_lon.begin(), b_lat.begin(), distance.begin());

  EXPECT_EQ(expected, distance);
  EXPECT_EQ(1, std::distance(distance.begin(), distance_end));
}

TYPED_TEST(HaversineTest, NegativeRadius)
{
  using T = TypeParam;

  auto a_lon = rmm::device_vector<T>{1, 0};
  auto a_lat = rmm::device_vector<T>{1, 0};
  auto b_lon = rmm::device_vector<T>{1, 0};
  auto b_lat = rmm::device_vector<T>{1, 0};

  auto distance = rmm::device_vector<T>{1, -1};
  auto expected = rmm::device_vector<T>{1, 0};

  EXPECT_THROW(cuspatial::haversine_distance(a_lon.begin(),
                                             a_lon.end(),
                                             a_lat.begin(),
                                             b_lon.begin(),
                                             b_lat.begin(),
                                             distance.begin(),
                                             T{-10}),
               cuspatial::logic_error);
}

TYPED_TEST(HaversineTest, EquivalentPoints)
{
  using T = TypeParam;

  auto h_a_lon    = std::vector<T>({-180, 180});
  auto h_a_lat    = std::vector<T>({0, 30});
  auto h_b_lon    = std::vector<T>({180, -180});
  auto h_b_lat    = std::vector<T>({0, 30});
  auto h_expected = std::vector<T>({1.5604449514735574e-12, 1.3513849691832763e-12});

  auto a_lon = rmm::device_vector<T>{h_a_lon};
  auto a_lat = rmm::device_vector<T>{h_a_lat};
  auto b_lon = rmm::device_vector<T>{h_b_lon};
  auto b_lat = rmm::device_vector<T>{h_b_lat};

  auto distance = rmm::device_vector<T>{2, -1};
  auto expected = rmm::device_vector<T>{h_expected};

  auto distance_end = cuspatial::haversine_distance(
    a_lon.begin(), a_lon.end(), a_lat.begin(), b_lon.begin(), b_lat.begin(), distance.begin());

  EXPECT_EQ(expected, distance);
  EXPECT_EQ(2, std::distance(distance.begin(), distance_end));
}
