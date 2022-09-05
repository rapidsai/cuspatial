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
#include <cuspatial/experimental/coordinate_transform.cuh>

#include <rmm/device_vector.hpp>

#include <gtest/gtest.h>

#include <thrust/iterator/transform_iterator.h>

template <typename T>
struct LonLatToCartesianTest : public ::testing::Test {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LonLatToCartesianTest, TestTypes);

TYPED_TEST(LonLatToCartesianTest, Empty)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_point_lonlat = std::vector<Loc>{};
  auto h_expected     = std::vector<Cart>{};

  auto point_lonlat = rmm::device_vector<Loc>{};
  auto expected     = rmm::device_vector<Cart>{};

  auto xy_output = rmm::device_vector<Cart>{};

  auto xy_end = cuspatial::lonlat_to_cartesian(
    point_lonlat.begin(), point_lonlat.end(), xy_output.begin(), origin);

  EXPECT_EQ(expected, xy_output);
  EXPECT_EQ(0, std::distance(xy_output.begin(), xy_end));
}

TYPED_TEST(LonLatToCartesianTest, Single)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_point_lonlat = std::vector<Loc>({{-90.664973, 42.493894}});
  auto h_expected     = std::vector<Cart>({{-0.01126195531216838, -0.21375777777718794}});

  auto point_lonlat = rmm::device_vector<Loc>{h_point_lonlat};
  auto expected     = rmm::device_vector<Cart>{h_expected};

  auto xy_output = rmm::device_vector<Cart>(1);

  auto xy_end = cuspatial::lonlat_to_cartesian(
    point_lonlat.begin(), point_lonlat.end(), xy_output.begin(), origin);

  EXPECT_EQ(expected, xy_output);
  EXPECT_EQ(1, std::distance(xy_output.begin(), xy_end));
}

TYPED_TEST(LonLatToCartesianTest, Extremes)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{0, 0};

  auto h_points_lonlat = std::vector<Loc>(
    {{0.0, -90.0}, {0.0, 90.0}, {-180.0, 0.0}, {180.0, 0.0}, {45.0, 0.0}, {-180.0, -90.0}});
  auto h_expected = std::vector<Cart>({{0.0, 10000.0},
                                       {0.0, -10000.0},
                                       {20000.0, 0.0},
                                       {-20000.0, 0.0},
                                       {-5000.0, 0.0},
                                       {14142.13562373095192015, 10000.0}});

  auto points_lonlat = rmm::device_vector<Loc>{h_points_lonlat};
  auto expected      = rmm::device_vector<Cart>{h_expected};

  auto xy_output = rmm::device_vector<Cart>(6, Cart{-1, -1});

  auto xy_end = cuspatial::lonlat_to_cartesian(
    points_lonlat.begin(), points_lonlat.end(), xy_output.begin(), origin);

  EXPECT_EQ(expected, xy_output);
  EXPECT_EQ(6, std::distance(xy_output.begin(), xy_end));
}

TYPED_TEST(LonLatToCartesianTest, Multiple)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_points_lonlat = std::vector<Loc>({{-90.664973, 42.493894},
                                           {-90.665393, 42.491520},
                                           {-90.664976, 42.491420},
                                           {-90.664537, 42.493823}});
  auto h_expected      = std::vector<Cart>({
    {-0.01126195531216838, -0.21375777777718794},
    {0.02314864865181343, 0.05002000000015667},
    {-0.01101638630252916, 0.06113111111163663},
    {-0.04698301003584082, -0.20586888888847929},
  });

  auto points_lonlat = rmm::device_vector<Loc>{h_points_lonlat};
  auto expected      = rmm::device_vector<Cart>{h_expected};

  auto xy_output = rmm::device_vector<Cart>(4, Cart{-1, -1});

  auto xy_end = cuspatial::lonlat_to_cartesian(
    points_lonlat.begin(), points_lonlat.end(), xy_output.begin(), origin);

  EXPECT_EQ(expected, xy_output);
  EXPECT_EQ(4, std::distance(xy_output.begin(), xy_end));
}

TYPED_TEST(LonLatToCartesianTest, OriginOutOfBounds)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-181, -91};

  auto h_point_lonlat = std::vector<Loc>{};
  auto h_expected     = std::vector<Cart>{};

  auto point_lonlat = rmm::device_vector<Loc>{};
  auto expected     = rmm::device_vector<Cart>{};

  auto xy_output = rmm::device_vector<Cart>{};

  EXPECT_THROW(cuspatial::lonlat_to_cartesian(
                 point_lonlat.begin(), point_lonlat.end(), xy_output.begin(), origin),
               cuspatial::logic_error);
}

template <typename T>
struct identity_xform {
  using Location = cuspatial::vec_2d<T>;
  __device__ Location operator()(Location const& loc) { return loc; };
};

// This test verifies that fancy iterators can be passed by using a pass-through transform_iterator
TYPED_TEST(LonLatToCartesianTest, TransformIterator)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_points_lonlat = std::vector<Loc>({{-90.664973, 42.493894},
                                           {-90.665393, 42.491520},
                                           {-90.664976, 42.491420},
                                           {-90.664537, 42.493823}});
  auto h_expected      = std::vector<Cart>({
    {-0.01126195531216838, -0.21375777777718794},
    {0.02314864865181343, 0.05002000000015667},
    {-0.01101638630252916, 0.06113111111163663},
    {-0.04698301003584082, -0.20586888888847929},
  });

  auto points_lonlat = rmm::device_vector<Loc>{h_points_lonlat};
  auto expected      = rmm::device_vector<Cart>{h_expected};

  auto xy_output = rmm::device_vector<Cart>(4, Cart{-1, -1});

  auto xform_begin = thrust::make_transform_iterator(points_lonlat.begin(), identity_xform<T>{});
  auto xform_end   = thrust::make_transform_iterator(points_lonlat.end(), identity_xform<T>{});

  auto xy_end = cuspatial::lonlat_to_cartesian(xform_begin, xform_end, xy_output.begin(), origin);

  EXPECT_EQ(expected, xy_output);
  EXPECT_EQ(4, std::distance(xy_output.begin(), xy_end));
}
