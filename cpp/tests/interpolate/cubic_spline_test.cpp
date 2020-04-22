/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <sys/time.h>
#include <time.h>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cuspatial/cubic_spline.hpp>
#include <cuspatial/error.hpp>
#include <string>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

struct CubicSplineTest : public cudf::test::BaseFixture {};

auto get_d_expect() {
  std::vector<float> d3_expect{{0.5, -0.5, -0.5, 0.5}};
  std::vector<float> d2_expect{{0, 3, 3, -6}};
  std::vector<float> d1_expect{{-1.5, -4.5, -4.5, 22.5}};
  std::vector<float> d0_expect{{3, 4, 4, -23}};
  return std::vector<std::vector<float>>{
      {d3_expect, d2_expect, d1_expect, d0_expect}};
}

TEST_F(CubicSplineTest, test_coefficients_single) {
  cudf::test::fixed_width_column_wrapper<float> t_column{{0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> y_column{{3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, y_column,
                                                     ids_column, prefix_column);

  auto d_expect = get_d_expect();
  for (unsigned int i = 0; i < d_expect.size(); ++i) {
    cudf::test::expect_columns_equivalent(
        splines->get_column(i), cudf::test::fixed_width_column_wrapper<float>(
                                    d_expect[i].begin(), d_expect[i].end()));
  }
}

TEST_F(CubicSplineTest, test_coefficients_full) {
  cudf::test::fixed_width_column_wrapper<float> t_column{
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> y_column{
      {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0, 1, 2}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5, 10, 15}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, y_column,
                                                     ids_column, prefix_column);

  auto d_expect = get_d_expect();
  for (unsigned int i = 0; i < d_expect.size(); ++i) {
    auto h_expected_col = d_expect[i];
    auto h_actual_col =
        cudf::test::to_host<float>(splines->get_column(i)).first;
    for (unsigned int j = 0; j < h_actual_col.size(); ++j) {
      EXPECT_EQ(h_expected_col[j % h_expected_col.size()], h_actual_col[j]);
    }
  }
}

TEST_F(CubicSplineTest, test_interpolate_single) {
  cudf::test::fixed_width_column_wrapper<float> t_column{{0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> x_column{{3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{{0, 0, 0, 0, 0}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, x_column,
                                                     ids_column, prefix_column);

  auto interpolates = cuspatial::cubicspline_interpolate(
      t_column, point_ids_column, prefix_column, t_column, splines->view());

  cudf::test::expect_columns_equivalent(
      *interpolates,
      cudf::test::fixed_width_column_wrapper<float>{{3, 2, 3, 4, 3}});
}

TEST_F(CubicSplineTest, test_interpolate_full) {
  cudf::test::fixed_width_column_wrapper<float> t_column{
      {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> x_column{
      {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0, 1, 2}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5, 10, 15}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, x_column,
                                                     ids_column, prefix_column);

  auto interpolates = cuspatial::cubicspline_interpolate(
      t_column, point_ids_column, prefix_column, t_column, splines->view());

  cudf::test::expect_columns_equivalent(
      *interpolates, cudf::test::fixed_width_column_wrapper<float>{
                         {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}});
}
