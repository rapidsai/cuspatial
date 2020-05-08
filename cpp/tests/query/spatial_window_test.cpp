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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cuspatial/spatial_window.hpp>

template <typename T>
struct SpatialWindowTest : public cudf::test::BaseFixture {};

using TestTypes = cudf::test::Types<float, double>;
TYPED_TEST_CASE(SpatialWindowTest, TestTypes);

TYPED_TEST(SpatialWindowTest, toy_test) {
  using T = TypeParam;
  // assuming x/y are in the unit of killometers (km);

  auto points_x = cudf::test::fixed_width_column_wrapper<T>(
      {1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0});
  auto points_y = cudf::test::fixed_width_column_wrapper<T>(
      {0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0});

  auto expected_points_x =
      cudf::test::fixed_width_column_wrapper<T>({3.0, 5.0, 2.0});
  auto expected_points_y =
      cudf::test::fixed_width_column_wrapper<T>({2.0, 3.0, 5.0});

  auto result = cuspatial::points_in_spatial_window(1.5, 1.5, 5.5, 5.5,
                                                    points_x, points_y);

  cudf::test::expect_columns_equivalent(result.first->view(), expected_points_x,
                                        true);
  cudf::test::expect_columns_equivalent(result.second->view(),
                                        expected_points_y, true);
}
