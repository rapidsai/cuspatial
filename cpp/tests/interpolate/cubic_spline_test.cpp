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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cuspatial/cubic_spline.hpp>
#include <cuspatial/detail/cubic_spline.hpp>
#include <cuspatial/error.hpp>

#include <string>
#include <sys/time.h>
#include <time.h>

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

  auto expected =
    cudf::table_view{{detail3_expected, detail2_expected, detail1_expected, detail0_expected}};

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*splines, expected);
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

  auto expected =
    cudf::table_view{{detail3_expected, detail2_expected, detail1_expected, detail0_expected}};

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*splines, expected);
}

TEST_F(CubicSplineTest, test_interpolate_between_control_points)
{
  cudf::test::fixed_width_column_wrapper<float> t_column{
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> new_column{
    {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 0.0, 0.5, 1.0, 1.5, 2.0,
     2.5, 3.0, 3.5, 4.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}};
  cudf::test::fixed_width_column_wrapper<float> x_column{
    {3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<int> ids_column{{0, 0, 1, 2}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5, 10, 15}};
  cudf::test::fixed_width_column_wrapper<int> old_point_ids_column{
    {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}};
  cudf::test::fixed_width_column_wrapper<int> new_point_ids_column{
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2}};

  auto splines = cuspatial::cubicspline_coefficients(t_column, x_column, ids_column, prefix_column);

  auto interpolants_new = cuspatial::cubicspline_interpolate(
    new_column, new_point_ids_column, prefix_column, t_column, *splines);

  auto gather_map = cudf::test::fixed_width_column_wrapper<int16_t>{
    0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26};
  auto interpolants_gather =
    cudf::gather(cudf::table_view{std::vector<cudf::column_view>{interpolants_new->view()}},
                 gather_map)
      ->get_column(0);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    interpolants_gather,
    cudf::test::fixed_width_column_wrapper<float>{{3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}});
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

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *interpolants, cudf::test::fixed_width_column_wrapper<float>{{3, 2, 3, 4, 3}});
}

TEST_F(CubicSplineTest, test_interpolate_at_control_points_full)
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

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *interpolants,
    cudf::test::fixed_width_column_wrapper<float>{{3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3}});
}

TEST_F(CubicSplineTest, test_parallel_search_single)
{
  cudf::test::fixed_width_column_wrapper<float> short_single{{0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{{0, 0, 0, 0, 0}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5}};

  auto indexes = cuspatial::detail::find_coefficient_indices(short_single,
                                                             point_ids_column,
                                                             prefix_column,
                                                             short_single,
                                                             rmm::cuda_stream_default,
                                                             this->mr());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*indexes,
                                      cudf::test::fixed_width_column_wrapper<int>{{0, 1, 2, 3, 3}});
}

TEST_F(CubicSplineTest, test_parallel_search_triple)
{
  cudf::test::fixed_width_column_wrapper<float> short_triple{
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{
    {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5, 10, 15}};

  auto indexes = cuspatial::detail::find_coefficient_indices(short_triple,
                                                             point_ids_column,
                                                             prefix_column,
                                                             short_triple,
                                                             rmm::cuda_stream_default,
                                                             this->mr());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *indexes,
    cudf::test::fixed_width_column_wrapper<int>{{0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11}});
}

TEST_F(CubicSplineTest, test_parallel_search_middle_single)
{
  cudf::test::fixed_width_column_wrapper<float> short_single{{0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> long_single{
    {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{{0, 0, 0, 0, 0, 0, 0, 0, 0}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5}};

  auto indexes = cuspatial::detail::find_coefficient_indices(long_single,
                                                             point_ids_column,
                                                             prefix_column,
                                                             short_single,
                                                             rmm::cuda_stream_default,
                                                             this->mr());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *indexes, cudf::test::fixed_width_column_wrapper<int>{{0, 0, 1, 1, 2, 2, 3, 3, 3}});
}

TEST_F(CubicSplineTest, test_parallel_search_middle_triple)
{
  cudf::test::fixed_width_column_wrapper<float> short_triple{
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> long_triple{
    {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 0.0, 0.5, 1.0, 1.5, 2.0,
     2.5, 3.0, 3.5, 4.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5, 10, 15}};

  auto indexes = cuspatial::detail::find_coefficient_indices(long_triple,
                                                             point_ids_column,
                                                             prefix_column,
                                                             short_triple,
                                                             rmm::cuda_stream_default,
                                                             this->mr());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *indexes,
    cudf::test::fixed_width_column_wrapper<int>{
      {0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 11}});
}

TEST_F(CubicSplineTest, test_parallel_search_single_end_values)
{
  cudf::test::fixed_width_column_wrapper<float> short_triple{{0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> long_triple{{4.0, 4.1, 4.5, 5.0, 10000.0}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{{0, 0, 0, 0, 0}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5}};

  auto indexes = cuspatial::detail::find_coefficient_indices(long_triple,
                                                             point_ids_column,
                                                             prefix_column,
                                                             short_triple,
                                                             rmm::cuda_stream_default,
                                                             this->mr());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*indexes,
                                      cudf::test::fixed_width_column_wrapper<int>{{3, 3, 3, 3, 3}});
}

TEST_F(CubicSplineTest, test_parallel_search_triple_end_values)
{
  cudf::test::fixed_width_column_wrapper<float> short_triple{
    {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};
  cudf::test::fixed_width_column_wrapper<float> long_triple{
    {4.0, 4.1, 4.5, 5.0, 10000.0, 4.0, 4.1, 4.5, 5.0, 10000.0, 4.0, 4.1, 4.5, 5.0, 10000.0}};
  cudf::test::fixed_width_column_wrapper<int> point_ids_column{
    {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}};
  cudf::test::fixed_width_column_wrapper<int> prefix_column{{0, 5}};

  auto indexes = cuspatial::detail::find_coefficient_indices(long_triple,
                                                             point_ids_column,
                                                             prefix_column,
                                                             short_triple,
                                                             rmm::cuda_stream_default,
                                                             this->mr());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*indexes,
                                      cudf::test::fixed_width_column_wrapper<int>{
                                        {3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 11, 11, 11, 11, 11}});
}
