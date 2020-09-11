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

#include <cudf/utilities/test/base_fixture.hpp>
#include <cudf/utilities/test/column_utilities.hpp>
#include <cudf/utilities/test/column_wrapper.hpp>
#include <cudf/utilities/test/cudf_gtest.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/shapefile_reader.hpp>

#include <cstdlib>

using namespace cudf::test;

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
  auto shape_filename  = get_shapefile_path(shapefile_name);
  auto polygon_columns = cuspatial::read_polygon_shapefile(shape_filename);

  auto expected_poly_offsets  = wrapper<cudf::size_type>(poly_offsets.begin(), poly_offsets.end());
  auto expected_ring_offsets  = wrapper<cudf::size_type>(ring_offsets.begin(), ring_offsets.end());
  auto expected_poly_point_xs = wrapper<double>(xs.begin(), xs.end());
  auto expected_poly_point_ys = wrapper<double>(ys.begin(), ys.end());

  expect_columns_equivalent(expected_poly_offsets, polygon_columns.at(0)->view(), true);
  expect_columns_equivalent(expected_ring_offsets, polygon_columns.at(1)->view(), true);
  expect_columns_equivalent(expected_poly_point_xs, polygon_columns.at(2)->view(), true);
  expect_columns_equivalent(expected_poly_point_ys, polygon_columns.at(3)->view(), true);
}

struct PolygonShapefileReaderTest : public BaseFixture {
};

TEST_F(PolygonShapefileReaderTest, NonExistentFile)
{
  auto shape_filename = get_shapefile_path("non_exist.shp");
  EXPECT_THROW(cuspatial::read_polygon_shapefile(shape_filename), cuspatial::logic_error);
}

TEST_F(PolygonShapefileReaderTest, ZeroPolygons) { test("empty_poly.shp", {}, {}, {}, {}); }

TEST_F(PolygonShapefileReaderTest, OnePolygon)
{
  test("one_poly.shp", {0}, {0}, {-10, 5, 5, -10, -10}, {-10, -10, 5, 5, -10});
}

TEST_F(PolygonShapefileReaderTest, TwoPolygons)
{
  test("two_polys.shp",
       {0, 1},
       {0, 5},
       {-10, 5, 5, -10, -10, 0, 10, 10, 0, 0},
       {-10, -10, 5, 5, -10, 0, 0, 10, 10, 0});
}
