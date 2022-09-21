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

#include <cuspatial/error.hpp>
#include <cuspatial/point_in_polygon.hpp>
#include <cuspatial/shapefile_reader.hpp>

#include <rmm/device_vector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cstdlib>

using namespace cudf::test;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

std::string get_shapefile_path(std::string filename)
{
  const char* cuspatial_home = std::getenv("CUSPATIAL_HOME");
  CUSPATIAL_EXPECTS(cuspatial_home != nullptr, "CUSPATIAL_HOME environmental variable must be set");
  return std::string(cuspatial_home) + std::string("/test_fixtures/shapefiles/") +
         std::string(filename);
}

template <typename T>
using wrapper = fixed_width_column_wrapper<T>;

void test(std::string const& shapefile_name,
          std::vector<cudf::size_type> poly_offsets,
          std::vector<cudf::size_type> ring_offsets,
          std::vector<double> xs,
          std::vector<double> ys)
{
  auto shape_filename = get_shapefile_path(shapefile_name);
  auto polygon_columns =
    cuspatial::read_polygon_shapefile(shape_filename, cuspatial::winding_order::COUNTER_CLOCKWISE);

  auto expected_poly_offsets  = wrapper<cudf::size_type>(poly_offsets.begin(), poly_offsets.end());
  auto expected_ring_offsets  = wrapper<cudf::size_type>(ring_offsets.begin(), ring_offsets.end());
  auto expected_poly_point_xs = wrapper<double>(xs.begin(), xs.end());
  auto expected_poly_point_ys = wrapper<double>(ys.begin(), ys.end());

  expect_columns_equivalent(expected_poly_offsets, polygon_columns.at(0)->view(), verbosity);
  expect_columns_equivalent(expected_ring_offsets, polygon_columns.at(1)->view(), verbosity);
  expect_columns_equivalent(expected_poly_point_xs, polygon_columns.at(2)->view(), verbosity);
  expect_columns_equivalent(expected_poly_point_ys, polygon_columns.at(3)->view(), verbosity);
}

void test_reverse(std::string const& shapefile_name,
                  std::vector<cudf::size_type> poly_offsets,
                  std::vector<cudf::size_type> ring_offsets,
                  std::vector<double> xs,
                  std::vector<double> ys)
{
  auto shape_filename = get_shapefile_path(shapefile_name);
  auto polygon_columns =
    cuspatial::read_polygon_shapefile(shape_filename, cuspatial::winding_order::CLOCKWISE);

  auto expected_poly_offsets  = wrapper<cudf::size_type>(poly_offsets.begin(), poly_offsets.end());
  auto expected_ring_offsets  = wrapper<cudf::size_type>(ring_offsets.begin(), ring_offsets.end());
  auto expected_poly_point_xs = wrapper<double>(xs.begin(), xs.end());
  auto expected_poly_point_ys = wrapper<double>(ys.begin(), ys.end());

  expect_columns_equivalent(expected_poly_offsets, polygon_columns.at(0)->view(), verbosity);
  expect_columns_equivalent(expected_ring_offsets, polygon_columns.at(1)->view(), verbosity);
  expect_columns_equivalent(expected_poly_point_xs, polygon_columns.at(2)->view(), verbosity);
  expect_columns_equivalent(expected_poly_point_ys, polygon_columns.at(3)->view(), verbosity);
}

struct PolygonShapefileReaderTest : public BaseFixture {
};

TEST_F(PolygonShapefileReaderTest, NonExistentFile)
{
  auto shape_filename = get_shapefile_path("non_exist.shp");
  EXPECT_THROW(cuspatial::read_polygon_shapefile(shape_filename), cuspatial::logic_error);
}

TEST_F(PolygonShapefileReaderTest, ZeroPolygons) { test("empty_poly.shp", {}, {}, {}, {}); }

TEST_F(PolygonShapefileReaderTest, OnePolygonReversed)
{
  test_reverse("one_poly.shp", {0}, {0}, {-10, 5, 5, -10, -10}, {-10, -10, 5, 5, -10});
}

TEST_F(PolygonShapefileReaderTest, OnePolygon)
{
  test("one_poly.shp", {0}, {0}, {-10, -10, 5, 5, -10}, {-10, 5, 5, -10, -10});
}

TEST_F(PolygonShapefileReaderTest, TwoPolygons)
{
  test("two_polys.shp",
       {0, 1},
       {0, 5},
       {-10, -10, 5, 5, -10, 0, 0, 10, 10, 0},
       {-10, 5, 5, -10, -10, 0, 10, 10, 0, 0});
}

TEST_F(PolygonShapefileReaderTest, OnePointInPolygon)
{
  auto shape_filename  = get_shapefile_path("one_poly.shp");
  auto polygon_columns = cuspatial::read_polygon_shapefile(shape_filename);

  auto polygons = polygon_columns.at(0)->view();
  auto rings    = polygon_columns.at(1)->view();
  auto xs       = polygon_columns.at(2)->view();
  auto ys       = polygon_columns.at(3)->view();
  fixed_width_column_wrapper<double> test_xs({0.0});
  fixed_width_column_wrapper<double> test_ys({0.0});
  fixed_width_column_wrapper<int32_t> expected({true});

  auto ret = cuspatial::point_in_polygon(test_xs, test_ys, polygons, rings, xs, ys);

  expect_columns_equivalent(ret->view(), expected);
}
