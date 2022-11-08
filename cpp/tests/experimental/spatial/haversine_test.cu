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

#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/haversine.cuh>

#include <rmm/device_vector.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <gtest/gtest.h>

template <typename T>
struct HaversineTest : public ::testing::Test {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(HaversineTest, TestTypes);

TYPED_TEST(HaversineTest, Empty)
{
  using T        = TypeParam;
  using Location = cuspatial::vec_2d<T>;

  auto a_lonlat = rmm::device_vector<Location>{};
  auto b_lonlat = rmm::device_vector<Location>{};

  auto distance = rmm::device_vector<T>{};
  auto expected = rmm::device_vector<T>{};

  auto distance_end = cuspatial::haversine_distance(
    a_lonlat.begin(), a_lonlat.end(), b_lonlat.begin(), distance.begin());

  cuspatial::test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(0, std::distance(distance.begin(), distance_end));
}

TYPED_TEST(HaversineTest, Zero)
{
  using T        = TypeParam;
  using Location = cuspatial::vec_2d<T>;
  using LocVec   = std::vector<Location>;

  auto a_lonlat = rmm::device_vector<Location>(1, Location{0, 0});
  auto b_lonlat = rmm::device_vector<Location>(1, Location{0, 0});

  auto distance = rmm::device_vector<T>{1, -1};
  auto expected = rmm::device_vector<T>{1, 0};

  auto distance_end = cuspatial::haversine_distance(
    a_lonlat.begin(), a_lonlat.end(), b_lonlat.begin(), distance.begin());

  cuspatial::test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(1, std::distance(distance.begin(), distance_end));
}

TYPED_TEST(HaversineTest, NegativeRadius)
{
  using T        = TypeParam;
  using Location = cuspatial::vec_2d<T>;
  using LocVec   = std::vector<Location>;

  auto a_lonlat = rmm::device_vector<Location>(LocVec({Location{1, 1}, Location{0, 0}}));
  auto b_lonlat = rmm::device_vector<Location>(LocVec({Location{1, 1}, Location{0, 0}}));

  auto distance = rmm::device_vector<T>{1, -1};
  auto expected = rmm::device_vector<T>{1, 0};

  EXPECT_THROW(cuspatial::haversine_distance(
                 a_lonlat.begin(), a_lonlat.end(), b_lonlat.begin(), distance.begin(), T{-10}),
               cuspatial::logic_error);
}

TYPED_TEST(HaversineTest, EquivalentPoints)
{
  using T        = TypeParam;
  using Location = cuspatial::vec_2d<T>;

  auto h_a_lonlat = std::vector<Location>({{-180, 0}, {180, 30}});
  auto h_b_lonlat = std::vector<Location>({{180, 0}, {-180, 30}});

  auto h_expected = std::vector<T>({1.5604449514735574e-12, 1.3513849691832763e-12});

  auto a_lonlat = rmm::device_vector<Location>{h_a_lonlat};
  auto b_lonlat = rmm::device_vector<Location>{h_b_lonlat};

  auto distance = rmm::device_vector<T>{2, -1};
  auto expected = rmm::device_vector<T>{h_expected};

  auto distance_end = cuspatial::haversine_distance(
    a_lonlat.begin(), a_lonlat.end(), b_lonlat.begin(), distance.begin());

  cuspatial::test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(2, std::distance(distance.begin(), distance_end));
}

template <typename T>
struct identity_xform {
  using Location = cuspatial::vec_2d<T>;
  __device__ Location operator()(Location const& loc) { return loc; };
};

// This test verifies that fancy iterators can be passed by using a pass-through transform_iterator
TYPED_TEST(HaversineTest, TransformIterator)
{
  using T        = TypeParam;
  using Location = cuspatial::vec_2d<T>;

  auto h_a_lonlat = std::vector<Location>({{-180, 0}, {180, 30}});
  auto h_b_lonlat = std::vector<Location>({{180, 0}, {-180, 30}});

  auto h_expected = std::vector<T>({1.5604449514735574e-12, 1.3513849691832763e-12});

  auto a_lonlat = rmm::device_vector<Location>{h_a_lonlat};
  auto b_lonlat = rmm::device_vector<Location>{h_b_lonlat};

  auto distance = rmm::device_vector<T>{2, -1};
  auto expected = rmm::device_vector<T>{h_expected};

  auto xform_begin = thrust::make_transform_iterator(a_lonlat.begin(), identity_xform<T>{});
  auto xform_end   = thrust::make_transform_iterator(a_lonlat.end(), identity_xform<T>{});

  auto distance_end =
    cuspatial::haversine_distance(xform_begin, xform_end, b_lonlat.begin(), distance.begin());

  cuspatial::test::expect_vector_equivalent(expected, distance);
  EXPECT_EQ(2, std::distance(distance.begin(), distance_end));
}
