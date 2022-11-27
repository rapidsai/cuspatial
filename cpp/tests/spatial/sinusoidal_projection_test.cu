/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cuspatial/projection.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <type_traits>

using namespace cudf::test;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
struct SinusoidalProjectionTest : public BaseFixture {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = Types<double>;
TYPED_TEST_CASE(SinusoidalProjectionTest, TestTypes);

TYPED_TEST(SinusoidalProjectionTest, Single)
{
  using T         = TypeParam;
  auto camera_lon = -90.66511046;
  auto camera_lat = 42.49197018;
  auto point_lon  = fixed_width_column_wrapper<T>({-90.664973});
  auto point_lat  = fixed_width_column_wrapper<T>({42.493894});

  auto res_pair = cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat);

  auto expected_lon = fixed_width_column_wrapper<T>({-0.01126195531216838});
  auto expected_lat = fixed_width_column_wrapper<T>({-0.21375777777718794});

  expect_columns_equivalent(expected_lon, res_pair.first->view(), verbosity);
  expect_columns_equivalent(expected_lat, res_pair.second->view(), verbosity);
}

TYPED_TEST(SinusoidalProjectionTest, Extremes)
{
  using T         = TypeParam;
  auto camera_lon = 0;
  auto camera_lat = 0;
  auto point_lon  = fixed_width_column_wrapper<T>({0.0, 0.0, -180.0, 180.0, 45.0, -180.0});
  auto point_lat  = fixed_width_column_wrapper<T>({-90.0, 90.0, 0.0, 0.0, 0.0, -90.0});

  auto res_pair = cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat);

  auto expected_lon =
    fixed_width_column_wrapper<T>({0.0, 0.0, 20000.0, -20000.0, -5000.0, 14142.13562373095192015});
  auto expected_lat = fixed_width_column_wrapper<T>({10000.0, -10000.0, 0.0, 0.0, 0.0, 10000.0});

  expect_columns_equivalent(expected_lon, res_pair.first->view(), verbosity);
  expect_columns_equivalent(expected_lat, res_pair.second->view(), verbosity);
}

TYPED_TEST(SinusoidalProjectionTest, Multiple)
{
  using T         = TypeParam;
  auto camera_lon = -90.66511046;
  auto camera_lat = 42.49197018;
  auto point_lon  = fixed_width_column_wrapper<T>({-90.664973, -90.665393, -90.664976, -90.664537});
  auto point_lat  = fixed_width_column_wrapper<T>({42.493894, 42.491520, 42.491420, 42.493823});

  auto res_pair = cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat);

  auto expected_lon = fixed_width_column_wrapper<T>(
    {-0.01126195531216838, 0.02314864865181343, -0.01101638630252916, -0.04698301003584082});
  auto expected_lat = fixed_width_column_wrapper<T>(
    {-0.21375777777718794, 0.05002000000015667, 0.06113111111163663, -0.20586888888847929});

  expect_columns_equivalent(expected_lon, res_pair.first->view(), verbosity);
  expect_columns_equivalent(expected_lat, res_pair.second->view(), verbosity);
}

TYPED_TEST(SinusoidalProjectionTest, Empty)
{
  using T         = TypeParam;
  auto camera_lon = -90.66511046;
  auto camera_lat = 42.49197018;
  auto point_lon  = fixed_width_column_wrapper<T>({});
  auto point_lat  = fixed_width_column_wrapper<T>({});

  auto res_pair = cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat);

  auto expected_lon = fixed_width_column_wrapper<T>({});
  auto expected_lat = fixed_width_column_wrapper<T>({});

  expect_columns_equivalent(expected_lon, res_pair.first->view(), verbosity);
  expect_columns_equivalent(expected_lat, res_pair.second->view(), verbosity);
}

TYPED_TEST(SinusoidalProjectionTest, NullableNoNulls)
{
  using T         = TypeParam;
  auto camera_lon = -90.66511046;
  auto camera_lat = 42.49197018;
  auto point_lon  = fixed_width_column_wrapper<T>({-90.664973}, {1});
  auto point_lat  = fixed_width_column_wrapper<T>({42.493894}, {1});

  auto res_pair = cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat);

  auto expected_lon = fixed_width_column_wrapper<T>({-0.01126195531216838});
  auto expected_lat = fixed_width_column_wrapper<T>({-0.21375777777718794});

  expect_columns_equivalent(expected_lon, res_pair.first->view(), verbosity);
  expect_columns_equivalent(expected_lat, res_pair.second->view(), verbosity);
}

TYPED_TEST(SinusoidalProjectionTest, NullabilityMixedNoNulls)
{
  using T         = TypeParam;
  auto camera_lon = -90.66511046;
  auto camera_lat = 42.49197018;
  auto point_lon  = fixed_width_column_wrapper<T>({-90.664973});
  auto point_lat  = fixed_width_column_wrapper<T>({42.493894}, {1});

  auto res_pair = cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat);

  auto expected_lon = fixed_width_column_wrapper<T>({-0.01126195531216838});
  auto expected_lat = fixed_width_column_wrapper<T>({-0.21375777777718794});

  expect_columns_equivalent(expected_lon, res_pair.first->view(), verbosity);
  expect_columns_equivalent(expected_lat, res_pair.second->view(), verbosity);
}

TYPED_TEST(SinusoidalProjectionTest, NullableWithNulls)
{
  using T         = TypeParam;
  auto camera_lon = 0;
  auto camera_lat = 0;
  auto point_lon  = fixed_width_column_wrapper<T>({0}, {0});
  auto point_lat  = fixed_width_column_wrapper<T>({0}, {1});

  EXPECT_THROW(cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat),
               cuspatial::logic_error);
}

TYPED_TEST(SinusoidalProjectionTest, OriginOutOfBounds)
{
  using T         = TypeParam;
  auto camera_lon = -181;
  auto camera_lat = -91;
  auto point_lon  = fixed_width_column_wrapper<T>({0});
  auto point_lat  = fixed_width_column_wrapper<T>({0});

  EXPECT_THROW(cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat),
               cuspatial::logic_error);
}

TYPED_TEST(SinusoidalProjectionTest, MismatchType)
{
  auto camera_lon = 0;
  auto camera_lat = 0;
  auto point_lon  = fixed_width_column_wrapper<double>({0});
  auto point_lat  = fixed_width_column_wrapper<float>({0});

  EXPECT_THROW(cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat),
               cuspatial::logic_error);
}

TYPED_TEST(SinusoidalProjectionTest, MismatchSize)
{
  using T         = TypeParam;
  auto camera_lon = 0;
  auto camera_lat = 0;
  auto point_lon  = fixed_width_column_wrapper<T>({0, 0});
  auto point_lat  = fixed_width_column_wrapper<T>({0});

  EXPECT_THROW(cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat),
               cuspatial::logic_error);
}

template <typename T>
struct LatLonToCartesianUnsupportedTypesTest : public BaseFixture {
};

using UnsupportedTestTypes = RemoveIf<ContainedIn<Types<float, double>>, NumericTypes>;
TYPED_TEST_CASE(LatLonToCartesianUnsupportedTypesTest, UnsupportedTestTypes);

TYPED_TEST(LatLonToCartesianUnsupportedTypesTest, MismatchSize)
{
  using T         = TypeParam;
  auto camera_lon = 0;
  auto camera_lat = 0;
  auto point_lon  = fixed_width_column_wrapper<T>({0});
  auto point_lat  = fixed_width_column_wrapper<T>({0});

  EXPECT_THROW(cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat),
               cuspatial::logic_error);
}

template <typename T>
struct LatLonToCartesianUnsupportedChronoTypesTest : public BaseFixture {
};

TYPED_TEST_CASE(LatLonToCartesianUnsupportedChronoTypesTest, ChronoTypes);

TYPED_TEST(LatLonToCartesianUnsupportedChronoTypesTest, MismatchSize)
{
  using T         = TypeParam;
  using R         = typename T::rep;
  auto camera_lon = 0;
  auto camera_lat = 0;
  auto point_lon  = fixed_width_column_wrapper<T, R>({R{0}});
  auto point_lat  = fixed_width_column_wrapper<T, R>({R{0}});

  EXPECT_THROW(cuspatial::sinusoidal_projection(camera_lon, camera_lat, point_lon, point_lat),
               cuspatial::logic_error);
}
