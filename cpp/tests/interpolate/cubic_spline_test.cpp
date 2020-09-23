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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cuspatial/cubic_spline.hpp>
#include <cuspatial/error.hpp>

#include <sys/time.h>
#include <time.h>
#include <string>

struct CubicSplineTest : public cudf::test::BaseFixture {
};

TEST_F(CubicSplineTest, test_coefficients_single)
{
  cudf::test::fixed_width_column_wrapper<float> t_column{{0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> y_column{{3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, y_column, ids_column, prefix_column);

  cudf::test::fixed_width_column_wrapper<float> detail3_expected{{0.5, -0.5, -0.5, 0.5}};
  cudf::test::fixed_width_column_wrapper<float> detail2_expected{{0.0, 3.0, 3.0, -6.0}};
  cudf::test::fixed_width_column_wrapper<float> detail1_expected{{-1.5, -4.5, -4.5, 22.5}};
  cudf::test::fixed_width_column_wrapper<float> detail0_expected{{3.0, 4.0, 4.0, -23.0}};

  cudf::test::expect_tables_equivalent(
    *splines,
    cudf::table_view{{detail3_expected, detail2_expected, detail1_expected, detail0_expected}});
}

TEST_F(CubicSplineTest, test_coefficients_full)
{
  cudf::test::fixed_width_column_wrapper<float> t_column{
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> y_column{
    {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0, 1, 2}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5, 10, 15}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, y_column, ids_column, prefix_column);

  cudf::test::fixed_width_column_wrapper<float> detail3_expected{
    {0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5}};
  cudf::test::fixed_width_column_wrapper<float> detail2_expected{
    {0.0, 3.0, 3.0, -6.0, 0.0, 3.0, 3.0, -6.0, 0.0, 3.0, 3.0, -6.0}};
  cudf::test::fixed_width_column_wrapper<float> detail1_expected{
    {-1.5, -4.5, -4.5, 22.5, -1.5, -4.5, -4.5, 22.5, -1.5, -4.5, -4.5, 22.5}};
  cudf::test::fixed_width_column_wrapper<float> detail0_expected{
    {3.0, 4.0, 4.0, -23.0, 3.0, 4.0, 4.0, -23.0, 3.0, 4.0, 4.0, -23.0}};

  cudf::test::expect_tables_equivalent(
    *splines,
    cudf::table_view{{detail3_expected, detail2_expected, detail1_expected, detail0_expected}});
}

TEST_F(CubicSplineTest, test_interpolate_single)
{
  cudf::test::fixed_width_column_wrapper<float> t_column{{0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> x_column{{3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{{0, 0, 0, 0, 0}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, x_column, ids_column, prefix_column);

  auto interpolants = cuspatial::cubicspline_interpolate(
    t_column, point_ids_column, prefix_column, t_column, *splines);

  cudf::test::expect_columns_equivalent(
    *interpolants, cudf::test::fixed_width_column_wrapper<float>{{3, 2, 3, 4, 3}});
}

TEST_F(CubicSplineTest, test_interpolate_full)
{
  cudf::test::fixed_width_column_wrapper<float> t_column{
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> x_column{
    {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0, 1, 2}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5, 10, 15}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{
    {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, x_column, ids_column, prefix_column);

  auto interpolants = cuspatial::cubicspline_interpolate(
    t_column, point_ids_column, prefix_column, t_column, *splines);

  cudf::test::expect_columns_equivalent(
    *interpolants,
    cudf::test::fixed_width_column_wrapper<float>{{3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}});
}
