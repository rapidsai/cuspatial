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
#include <cuspatial/experimental/point_distance.cuh>
#include <cuspatial/utility/vec_2d.hpp>

#include <rmm/device_vector.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

#include <gtest/gtest.h>

namespace cuspatial {

template <typename T>
struct PairwisePointDistanceTest : public ::testing::Test {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PairwisePointDistanceTest, TestTypes);

TYPED_TEST(PairwisePointDistanceTest, Empty)
{
  using T         = TypeParam;
  using Cart2D    = cartesian_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  rmm::device_vector<Cart2D> points1{};
  rmm::device_vector<Cart2D> points2{};

  rmm::device_vector<T> expected{};
  rmm::device_vector<T> got(points1.size());

  pairwise_point_distance(points1.begin(), points1.end(), points2.begin(), got.begin());

  EXPECT_EQ(expected, got);
}

TYPED_TEST(PairwisePointDistanceTest, OnePair)
{
  using T         = TypeParam;
  using Cart2D    = cartesian_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  rmm::device_vector<Cart2D> points1{Cart2DVec{{1.0, 1.0}}};
  rmm::device_vector<Cart2D> points2{Cart2DVec{{0.0, 0.0}}};

  rmm::device_vector<T> expected{std::vector<T>{std::sqrt(T{2.0})}};
  rmm::device_vector<T> got(points1.size());

  pairwise_point_distance(points1.begin(), points1.end(), points2.begin(), got.begin());

  EXPECT_EQ(expected, got);
}

TYPED_TEST(PairwisePointDistanceTest, ManyRandom)
{
  using T         = TypeParam;
  using Cart2D    = cartesian_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  size_t num_points = 1000;

  thrust::minstd_rand rng;
  thrust::random::normal_distribution<T> norm_dist{-100.0, 100.0};

  rmm::device_vector<Cart2D> points1(num_points);
  auto rng_begin = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                   [rng, norm_dist] __device__(auto i) {
                                                     return Cart2D{norm_dist(rng), norm_dist(rng)};
                                                   });
  thrust::copy(rng_begin, rng_begin + num_points, points1.begin());

  // rmm::device_vector<Cart2D> points2{Cart2DVec{{0.0, 0.0}}};

  // rmm::device_vector<T> expected{std::vector<T>{std::sqrt(T{2.0})}};
  // rmm::device_vector<T> got(points1.size());

  // pairwise_point_distance(points1.begin(), points1.end(), points2.begin(), got.begin());

  // EXPECT_EQ(expected, got);
}

}  // namespace cuspatial
