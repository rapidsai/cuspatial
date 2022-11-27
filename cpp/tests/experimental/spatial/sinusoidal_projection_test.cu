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

#include <cuspatial/constants.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/sinusoidal_projection.cuh>
#include <cuspatial_test/vector_equality.hpp>

#include <rmm/device_vector.hpp>

#include <gtest/gtest.h>

#include <thrust/iterator/transform_iterator.h>

template <typename T>
inline T midpoint(T a, T b)
{
  return (a + b) / 2;
}

template <typename T>
inline T lon_to_x(T lon, T lat)
{
  return lon * cuspatial::EARTH_CIRCUMFERENCE_KM_PER_DEGREE *
         cos(lat * cuspatial::DEGREE_TO_RADIAN);
};

template <typename T>
inline T lat_to_y(T lat)
{
  return lat * cuspatial::EARTH_CIRCUMFERENCE_KM_PER_DEGREE;
};

template <typename T>
struct sinusoidal_projection_functor {
  using vec_2d = cuspatial::vec_2d<T>;

  sinusoidal_projection_functor(vec_2d origin) : _origin(origin) {}

  vec_2d operator()(vec_2d loc)
  {
    return vec_2d{lon_to_x(_origin.x - loc.x, midpoint(loc.y, _origin.y)),
                  lat_to_y(_origin.y - loc.y)};
  }

 private:
  vec_2d _origin{};
};

template <typename T>
struct SinusoidalProjectionTest : public ::testing::Test {
  using Vec = cuspatial::vec_2d<T>;

  void run_test(std::vector<Vec> const& h_lonlats, Vec const& origin)
  {
    auto h_expected = std::vector<Vec>(h_lonlats.size());

    std::transform(h_lonlats.begin(),
                   h_lonlats.end(),
                   h_expected.begin(),
                   sinusoidal_projection_functor(origin));

    auto lonlats = rmm::device_vector<Vec>{h_lonlats};

    auto xy_output = rmm::device_vector<Vec>(lonlats.size(), Vec{-1, -1});

    auto xy_end =
      cuspatial::sinusoidal_projection(lonlats.begin(), lonlats.end(), xy_output.begin(), origin);

    cuspatial::test::expect_vector_equivalent(h_expected, xy_output);
    EXPECT_EQ(h_expected.size(), std::distance(xy_output.begin(), xy_end));
  }
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(SinusoidalProjectionTest, TestTypes);

TYPED_TEST(SinusoidalProjectionTest, Empty)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_point_lonlat = std::vector<Loc>{};
  this->run_test(h_point_lonlat, origin);
}

TYPED_TEST(SinusoidalProjectionTest, Single)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_point_lonlat = std::vector<Loc>({{-90.664973, 42.493894}});
  this->run_test(h_point_lonlat, origin);
}

TYPED_TEST(SinusoidalProjectionTest, Extremes)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{0, 0};

  auto h_points_lonlat = std::vector<Loc>(
    {{0.0, -90.0}, {0.0, 90.0}, {-180.0, 0.0}, {180.0, 0.0}, {45.0, 0.0}, {-180.0, -90.0}});
  this->run_test(h_points_lonlat, origin);
}

TYPED_TEST(SinusoidalProjectionTest, Multiple)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_points_lonlat = std::vector<Loc>({{-90.664973, 42.493894},
                                           {-90.665393, 42.491520},
                                           {-90.664976, 42.491420},
                                           {-90.664537, 42.493823}});
  this->run_test(h_points_lonlat, origin);
}

TYPED_TEST(SinusoidalProjectionTest, OriginOutOfBounds)
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

  EXPECT_THROW(cuspatial::sinusoidal_projection(
                 point_lonlat.begin(), point_lonlat.end(), xy_output.begin(), origin),
               cuspatial::logic_error);
}

template <typename T>
struct identity_xform {
  using Location = cuspatial::vec_2d<T>;
  __device__ Location operator()(Location const& loc) { return loc; };
};

// This test verifies that fancy iterators can be passed by using a pass-through transform_iterator
TYPED_TEST(SinusoidalProjectionTest, TransformIterator)
{
  using T    = TypeParam;
  using Loc  = cuspatial::vec_2d<T>;
  using Cart = cuspatial::vec_2d<T>;

  auto origin = Loc{-90.66511046, 42.49197018};

  auto h_points_lonlat = std::vector<Loc>({{-90.664973, 42.493894},
                                           {-90.665393, 42.491520},
                                           {-90.664976, 42.491420},
                                           {-90.664537, 42.493823}});
  auto h_expected      = std::vector<Cart>(h_points_lonlat.size());

  std::transform(h_points_lonlat.begin(),
                 h_points_lonlat.end(),
                 h_expected.begin(),
                 sinusoidal_projection_functor(origin));

  auto points_lonlat = rmm::device_vector<Loc>{h_points_lonlat};
  auto expected      = rmm::device_vector<Cart>{h_expected};

  auto xy_output = rmm::device_vector<Cart>(4, Cart{-1, -1});

  auto xform_begin = thrust::make_transform_iterator(points_lonlat.begin(), identity_xform<T>{});
  auto xform_end   = thrust::make_transform_iterator(points_lonlat.end(), identity_xform<T>{});

  auto xy_end = cuspatial::sinusoidal_projection(xform_begin, xform_end, xy_output.begin(), origin);

  EXPECT_EQ(expected, xy_output);
  EXPECT_EQ(4, std::distance(xy_output.begin(), xy_end));
}
