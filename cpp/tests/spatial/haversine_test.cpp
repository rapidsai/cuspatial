/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cuspatial/distance/haversine.hpp>
#include <cuspatial/error.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <type_traits>

using namespace cudf::test;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
struct HaversineTest : public BaseFixture {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = Types<double>;
TYPED_TEST_CASE(HaversineTest, TestTypes);

TYPED_TEST(HaversineTest, Empty)
{
  using T = TypeParam;

  auto a_lon = fixed_width_column_wrapper<T>({});
  auto a_lat = fixed_width_column_wrapper<T>({});
  auto b_lon = fixed_width_column_wrapper<T>({});
  auto b_lat = fixed_width_column_wrapper<T>({});

  auto expected = fixed_width_column_wrapper<T>({});

  auto actual = cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view(), verbosity);
}

TYPED_TEST(HaversineTest, Zero)
{
  using T = TypeParam;

  auto a_lon = fixed_width_column_wrapper<T>({0});
  auto a_lat = fixed_width_column_wrapper<T>({0});
  auto b_lon = fixed_width_column_wrapper<T>({0});
  auto b_lat = fixed_width_column_wrapper<T>({0});

  auto expected = fixed_width_column_wrapper<T>({0});

  auto actual = cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view(), verbosity);
}

TYPED_TEST(HaversineTest, EquivalentPoints)
{
  using T = TypeParam;

  auto a_lon = fixed_width_column_wrapper<T>({-180, 180});
  auto a_lat = fixed_width_column_wrapper<T>({0, 30});
  auto b_lon = fixed_width_column_wrapper<T>({180, -180});
  auto b_lat = fixed_width_column_wrapper<T>({0, 30});

  auto expected = fixed_width_column_wrapper<T>({1.5604449514735574e-12, 1.3513849691832763e-12});

  auto actual = cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view(), verbosity);
}

TYPED_TEST(HaversineTest, MismatchSize)
{
  using T = TypeParam;

  auto a_lon = fixed_width_column_wrapper<T>({0});
  auto a_lat = fixed_width_column_wrapper<T>({0, 1});
  auto b_lon = fixed_width_column_wrapper<T>({0});
  auto b_lat = fixed_width_column_wrapper<T>({0});

  EXPECT_THROW(cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat), cuspatial::logic_error);
}

template <typename T>
struct HaversineUnsupportedTypesTest : public BaseFixture {
};

using UnsupportedTypes = RemoveIf<ContainedIn<Types<float, double>>, NumericTypes>;
TYPED_TEST_CASE(HaversineUnsupportedTypesTest, UnsupportedTypes);

TYPED_TEST(HaversineUnsupportedTypesTest, MismatchSize)
{
  using T = TypeParam;

  auto a_lon = fixed_width_column_wrapper<T>({0});
  auto a_lat = fixed_width_column_wrapper<T>({0});
  auto b_lon = fixed_width_column_wrapper<T>({0});
  auto b_lat = fixed_width_column_wrapper<T>({0});

  EXPECT_THROW(cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat), cuspatial::logic_error);
}

template <typename T>
struct HaversineUnsupportedChronoTypesTest : public BaseFixture {
};

TYPED_TEST_CASE(HaversineUnsupportedChronoTypesTest, ChronoTypes);

TYPED_TEST(HaversineUnsupportedChronoTypesTest, MismatchSize)
{
  using T = TypeParam;
  using R = typename T::rep;

  auto a_lon = fixed_width_column_wrapper<T, R>({R{0}});
  auto a_lat = fixed_width_column_wrapper<T, R>({R{0}});
  auto b_lon = fixed_width_column_wrapper<T, R>({R{0}});
  auto b_lat = fixed_width_column_wrapper<T, R>({R{0}});

  EXPECT_THROW(cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat), cuspatial::logic_error);
}
