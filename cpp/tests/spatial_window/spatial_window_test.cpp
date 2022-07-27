/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cuspatial/spatial_window.hpp>

#include <cudf/table/table.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <type_traits>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
struct SpatialRangeTest : public cudf::test::BaseFixture {
};

using TestTypes = cudf::test::Types<float, double>;

struct SpatialWindowErrorTest : public cudf::test::BaseFixture {
};

TEST_F(SpatialWindowErrorTest, TypeMismatch)
{
  auto points_x = cudf::test::fixed_width_column_wrapper<float>({1.0, 2.0, 3.0});
  auto points_y = cudf::test::fixed_width_column_wrapper<double>({0.0, 1.0, 2.0});

  EXPECT_THROW(
    auto result = cuspatial::points_in_spatial_window(1.5, 5.5, 1.5, 5.5, points_x, points_y),
    cuspatial::logic_error);
}

TEST_F(SpatialWindowErrorTest, SizeMismatch)
{
  auto points_x = cudf::test::fixed_width_column_wrapper<double>({1.0, 2.0, 3.0});
  auto points_y = cudf::test::fixed_width_column_wrapper<double>({0.0});

  EXPECT_THROW(
    auto result = cuspatial::points_in_spatial_window(1.5, 5.5, 1.5, 5.5, points_x, points_y),
    cuspatial::logic_error);
}

struct IsFloat {
  template <class T>
  using Call = std::is_floating_point<T>;
};

using NonFloatTypes = cudf::test::RemoveIf<IsFloat, cudf::test::NumericTypes>;

template <typename T>
struct SpatialWindowUnsupportedTypesTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SpatialWindowUnsupportedTypesTest, NonFloatTypes);

TYPED_TEST(SpatialWindowUnsupportedTypesTest, ShouldThrow)
{
  auto points_x = cudf::test::fixed_width_column_wrapper<TypeParam>({1.0, 2.0, 3.0});
  auto points_y = cudf::test::fixed_width_column_wrapper<TypeParam>({0.0, 1.0, 2.0});

  EXPECT_THROW(
    auto result = cuspatial::points_in_spatial_window(1.5, 5.5, 1.5, 5.5, points_x, points_y),
    cuspatial::logic_error);
}

template <typename T>
struct SpatialWindowUnsupportedChronoTypesTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SpatialWindowUnsupportedChronoTypesTest, cudf::test::ChronoTypes);

TYPED_TEST(SpatialWindowUnsupportedChronoTypesTest, ShouldThrow)
{
  using T       = TypeParam;
  using R       = typename T::rep;
  auto points_x = cudf::test::fixed_width_column_wrapper<T, R>({R{1}, R{2}, R{3}});
  auto points_y = cudf::test::fixed_width_column_wrapper<T, R>({R{0}, R{1}, R{2}});

  EXPECT_THROW(
    auto result = cuspatial::points_in_spatial_window(1.5, 5.5, 1.5, 5.5, points_x, points_y),
    cuspatial::logic_error);
}
