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
#include <cuspatial/experimental/points_in_range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <limits>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <type_traits>

#include <gtest/gtest.h>

template <typename T>
using Vec = cuspatial::vec_2d<T>;

template <typename T>
using VecVec = std::vector<Vec<T>>;

template <typename T>
using DeviceVecVec = rmm::device_vector<Vec<T>>;

template <typename T>
struct SpatialRangeTest : public testing::Test {
  void spatial_range_test(Vec<T> const& v1,
                          Vec<T> const& v2,
                          DeviceVecVec<T> const& points,
                          DeviceVecVec<T> const& expected_points)
  {
    auto result_size = cuspatial::count_points_in_range(v1, v2, points.begin(), points.end());

    EXPECT_EQ(result_size, expected_points.size());

    auto result_points = DeviceVecVec<T>(result_size);
    cuspatial::copy_points_in_range(v1, v2, points.begin(), points.end(), result_points.begin());

    cuspatial::test::expect_vector_equivalent(expected_points, result_points);
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(SpatialRangeTest, TestTypes);

TYPED_TEST(SpatialRangeTest, Empty)
{
  using T              = TypeParam;
  auto points          = DeviceVecVec<T>{};
  auto expected_points = DeviceVecVec<T>{};

  this->spatial_range_test(Vec<T>{1.5, 1.5}, Vec<T>{5.5, 5.5}, points, expected_points);
}

TYPED_TEST(SpatialRangeTest, SimpleTest)
{
  using T     = TypeParam;
  auto points = DeviceVecVec<T>(VecVec<T>({{1.0, 0.0},
                                           {2.0, 1.0},
                                           {3.0, 2.0},
                                           {5.0, 3.0},
                                           {7.0, 1.0},
                                           {1.0, 3.0},
                                           {2.0, 5.0},
                                           {3.0, 6.0},
                                           {6.0, 5.0},
                                           {0.0, 4.0},
                                           {3.0, 7.0},
                                           {6.0, 4.0}}));

  auto expected_points = DeviceVecVec<T>(VecVec<T>({{3.0, 2.0}, {5.0, 3.0}, {2.0, 5.0}}));

  this->spatial_range_test(Vec<T>{1.5, 1.5}, Vec<T>{5.5, 5.5}, points, expected_points);
}

// Test that ranges with min/max reversed still work
TYPED_TEST(SpatialRangeTest, ReversedRange)
{
  using T     = TypeParam;
  auto points = DeviceVecVec<T>(VecVec<T>({{1.0, 0.0},
                                           {2.0, 1.0},
                                           {3.0, 2.0},
                                           {5.0, 3.0},
                                           {7.0, 1.0},
                                           {1.0, 3.0},
                                           {2.0, 5.0},
                                           {3.0, 6.0},
                                           {6.0, 5.0},
                                           {0.0, 4.0},
                                           {3.0, 7.0},
                                           {6.0, 4.0}}));

  auto expected_points = DeviceVecVec<T>(VecVec<T>({{3.0, 2.0}, {5.0, 3.0}, {2.0, 5.0}}));

  this->spatial_range_test(Vec<T>{5.5, 5.5}, Vec<T>{1.5, 1.5}, points, expected_points);
}

TYPED_TEST(SpatialRangeTest, AllPointsInRange)
{
  using T     = TypeParam;
  auto points = DeviceVecVec<T>(VecVec<T>({{1.0, 0.0},
                                           {2.0, 1.0},
                                           {3.0, 2.0},
                                           {5.0, 3.0},
                                           {7.0, 1.0},
                                           {1.0, 3.0},
                                           {2.0, 5.0},
                                           {3.0, 6.0},
                                           {6.0, 5.0},
                                           {0.0, 4.0},
                                           {3.0, 7.0},
                                           {6.0, 4.0}}));

  auto expected_points = DeviceVecVec<T>(VecVec<T>({{1.0, 0.0},
                                                    {2.0, 1.0},
                                                    {3.0, 2.0},
                                                    {5.0, 3.0},
                                                    {7.0, 1.0},
                                                    {1.0, 3.0},
                                                    {2.0, 5.0},
                                                    {3.0, 6.0},
                                                    {6.0, 5.0},
                                                    {0.0, 4.0},
                                                    {3.0, 7.0},
                                                    {6.0, 4.0}}));

  this->spatial_range_test(Vec<T>{-10.0, -10.0}, Vec<T>{10.0, 10.0}, points, expected_points);
}

TYPED_TEST(SpatialRangeTest, PointsOnOrNearEdges)
{
  using T = TypeParam;

  Vec<T> v1 = {0.0, 0.0};
  Vec<T> v2 = {1.0, 1.0};

  auto eps   = std::numeric_limits<T>::epsilon();
  auto v_eps = Vec<T>{eps, eps};

  auto on_ll = v1;
  auto on_ul = Vec<T>{v1.x, v2.y};
  auto on_lr = Vec<T>{v2.x, v1.y};
  auto on_ur = v2;

  auto on_left   = Vec<T>{v1.x, 0.5};
  auto on_right  = Vec<T>{v2.x, 0.5};
  auto on_bottom = Vec<T>{0.5, v1.y};
  auto on_top    = Vec<T>{0.5, v2.y};

  auto in_ll     = on_ll + v_eps;
  auto in_ul     = on_ul + Vec<T>{eps, -eps};
  auto in_lr     = on_lr + Vec<T>{-eps, eps};
  auto in_ur     = on_ur - v_eps;
  auto in_left   = on_left + v_eps;
  auto in_right  = on_right - v_eps;
  auto in_bottom = on_bottom + v_eps;
  auto in_top    = on_top - v_eps;

  auto out_ll     = on_ll - v_eps;
  auto out_ul     = on_ul + Vec<T>{-eps, eps};
  auto out_lr     = on_lr + Vec<T>{eps, -eps};
  auto out_ur     = on_ur + v_eps;
  auto out_left   = on_left - v_eps;
  auto out_right  = on_right + v_eps;
  auto out_bottom = on_bottom - v_eps;
  auto out_top    = on_top + v_eps;

  auto points = DeviceVecVec<T>(
    VecVec<T>({on_ll,  on_ul,  on_lr,  on_ur,  on_left,  on_right,  on_bottom,  on_top,
               in_ll,  in_ul,  in_lr,  in_ur,  in_left,  in_right,  in_bottom,  in_top,
               out_ll, out_ul, out_lr, out_ur, out_left, out_right, out_bottom, out_top}));

  auto expected_points =
    DeviceVecVec<T>(VecVec<T>({in_ll, in_ul, in_lr, in_ur, in_left, in_right, in_bottom, in_top}));

  this->spatial_range_test(v1, v2, points, expected_points);
}
