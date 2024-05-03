/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial/bounding_boxes.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gtest/gtest.h>

#include <memory>

using T = float;

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;

struct QuadtreePointToNearestLinestringErrorTest : public ::testing::Test {
  auto prepare_test(cudf::column_view const& x,
                    cudf::column_view const& y,
                    cudf::column_view const& linestring_offsets,
                    cudf::column_view const& polygon_x,
                    cudf::column_view const& polygon_y,
                    cuspatial::vec_2d<T> v_min,
                    cuspatial::vec_2d<T> v_max,
                    T scale,
                    uint32_t max_depth,
                    uint32_t min_size,
                    T expansion_radius)
  {
    using namespace cudf::test;

    auto quadtree_pair = cuspatial::quadtree_on_points(
      x, y, v_min.x, v_max.x, v_min.y, v_max.y, scale, max_depth, min_size);

    auto& quadtree      = std::get<1>(quadtree_pair);
    auto& point_indices = std::get<0>(quadtree_pair);

    auto linestring_bboxes = cuspatial::linestring_bounding_boxes(
      linestring_offsets, polygon_x, polygon_y, expansion_radius);

    auto linestring_quadrant_pairs = cuspatial::join_quadtree_and_bounding_boxes(
      *quadtree, *linestring_bboxes, v_min.x, v_max.x, v_min.y, v_max.y, scale, max_depth);

    return std::make_tuple(std::move(quadtree),
                           std::move(point_indices),
                           std::move(linestring_bboxes),
                           std::move(linestring_quadrant_pairs));
  }

  void SetUp() override
  {
    using namespace cudf::test;

    cuspatial::vec_2d<T> v_min{0.0, 0.0};
    cuspatial::vec_2d<T> v_max{8.0, 8.0};
    double const scale{1.0};
    uint32_t const max_depth{3};
    uint32_t const min_size{12};
    double const expansion_radius{0.0};

    auto x_col =
      wrapper<T>({1.9804558865545805, 0.1895259128530169, 1.2591725716781235, 0.8178039499335275});
    auto y_col =
      wrapper<T>({1.3472225743317712, 0.5431061133894604, 0.1448705855995005, 0.8138440641113271});

    auto linestring_offsets_col = wrapper<std::int32_t>({0, 4, 10});
    auto linestring_x_col       = wrapper<T>({// ring 1
                                        2.488450,
                                        1.333584,
                                        3.460720,
                                        2.488450,
                                        // ring 2
                                        5.039823,
                                        5.561707,
                                        7.103516,
                                        7.190674,
                                        5.998939,
                                        5.039823});
    auto linestring_y_col       = wrapper<T>({// ring 1
                                        2.488450,
                                        1.333584,
                                        3.460720,
                                        2.488450,
                                        // ring 2
                                        5.039823,
                                        5.561707,
                                        7.103516,
                                        7.190674,
                                        5.998939,
                                        5.039823});

    std::tie(quadtree, point_indices, linestring_bboxes, linestring_quadrant_pairs) =
      prepare_test(x_col,
                   y_col,
                   linestring_offsets_col,
                   linestring_x_col,
                   linestring_y_col,
                   v_min,
                   v_max,
                   scale,
                   max_depth,
                   min_size,
                   expansion_radius);

    x                  = x_col.release();
    y                  = y_col.release();
    linestring_offsets = linestring_offsets_col.release();
    linestring_x       = linestring_x_col.release();
    linestring_y       = linestring_y_col.release();
  }

  void TearDown() override {}

  std::unique_ptr<cudf::column> x;
  std::unique_ptr<cudf::column> y;
  std::unique_ptr<cudf::column> linestring_offsets;
  std::unique_ptr<cudf::column> linestring_x;
  std::unique_ptr<cudf::column> linestring_y;
  std::unique_ptr<cudf::column> point_indices;
  std::unique_ptr<cudf::table> quadtree;
  std::unique_ptr<cudf::table> linestring_bboxes;
  std::unique_ptr<cudf::table> linestring_quadrant_pairs;
};

// test cudf::quadtree_point_in_polygon with empty inputs
TEST_F(QuadtreePointToNearestLinestringErrorTest, test_empty)
{
  // empty point data
  {
    auto empty_point_indices = wrapper<std::uint32_t>({});
    auto empty_x             = wrapper<T>({});
    auto empty_y             = wrapper<T>({});

    auto results = cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                   *quadtree,
                                                                   empty_point_indices,
                                                                   empty_x,
                                                                   empty_y,
                                                                   *linestring_offsets,
                                                                   *linestring_x,
                                                                   *linestring_y);

    auto expected_linestring_offset = wrapper<std::uint32_t>({});
    auto expected_point_offset      = wrapper<std::uint32_t>({});
    auto expected_distance          = wrapper<T>({});

    auto expected =
      cudf::table_view{{expected_linestring_offset, expected_point_offset, expected_distance}};

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *results);
  }

  // empty linestring data
  {
    auto empty_linestring_offsets = wrapper<std::uint32_t>({});
    auto empty_linestring_x       = wrapper<T>({});
    auto empty_linestring_y       = wrapper<T>({});

    auto results = cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                   *quadtree,
                                                                   *point_indices,
                                                                   *x,
                                                                   *y,
                                                                   empty_linestring_offsets,
                                                                   empty_linestring_x,
                                                                   empty_linestring_y);

    auto expected_linestring_offset = wrapper<std::uint32_t>({});
    auto expected_point_offset      = wrapper<std::uint32_t>({});
    auto expected_distance          = wrapper<T>({});

    auto expected =
      cudf::table_view{{expected_linestring_offset, expected_point_offset, expected_distance}};

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *results);
  }
}

TEST_F(QuadtreePointToNearestLinestringErrorTest, type_mismatch)
{
  // x/y type mismatch
  {
    auto x_col = wrapper<int32_t>({1, 2, 3, 4});
    auto y_col = wrapper<float>({1, 2, 3, 4});

    EXPECT_THROW(cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                 *quadtree,
                                                                 *point_indices,
                                                                 x_col,
                                                                 y_col,
                                                                 *linestring_offsets,
                                                                 *linestring_x,
                                                                 *linestring_y),
                 cuspatial::logic_error);
  }

  // linestring_x/linestring_y type mismatch
  {
    auto linestring_x_col = wrapper<int32_t>({1, 2, 3, 4});
    auto linestring_y_col = wrapper<float>({1, 2, 3, 4});

    EXPECT_THROW(cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                 *quadtree,
                                                                 *point_indices,
                                                                 *x,
                                                                 *y,
                                                                 *linestring_offsets,
                                                                 linestring_x_col,
                                                                 linestring_y_col),
                 cuspatial::logic_error);
  }

  // x / linestring_x type mismatch
  {
    auto x_col            = wrapper<int32_t>({1, 2, 3, 4});
    auto linestring_x_col = wrapper<float>({1, 2, 3, 4});

    EXPECT_THROW(cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                 *quadtree,
                                                                 *point_indices,
                                                                 x_col,
                                                                 *y,
                                                                 *linestring_offsets,
                                                                 linestring_x_col,
                                                                 *linestring_y),
                 cuspatial::logic_error);
  }
}

TEST_F(QuadtreePointToNearestLinestringErrorTest, size_mismatch)
{
  {
    auto linestring_offsets_col = wrapper<int32_t>({0, 4, 10});
    auto linestring_x           = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto linestring_y           = wrapper<float>({1, 2, 3, 4, 5, 6});

    EXPECT_THROW(cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                 *quadtree,
                                                                 *point_indices,
                                                                 *x,
                                                                 *y,
                                                                 linestring_offsets_col,
                                                                 linestring_x,
                                                                 linestring_y),
                 cuspatial::logic_error);
  }
  {
    auto point_indices = wrapper<int32_t>({0, 1, 2, 3, 4});
    auto x             = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto y             = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    EXPECT_THROW(cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                 *quadtree,
                                                                 point_indices,
                                                                 x,
                                                                 y,
                                                                 *linestring_offsets,
                                                                 *linestring_x,
                                                                 *linestring_y),
                 cuspatial::logic_error);
  }
  {
    auto point_indices = wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto x             = wrapper<float>({1, 2, 3, 4, 5});
    auto y             = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    EXPECT_THROW(cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                 *quadtree,
                                                                 point_indices,
                                                                 x,
                                                                 y,
                                                                 *linestring_offsets,
                                                                 *linestring_x,
                                                                 *linestring_y),
                 cuspatial::logic_error);
  }
  {
    auto point_indices = wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto x             = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto y             = wrapper<float>({1, 2, 3, 4, 5});

    EXPECT_THROW(cuspatial::quadtree_point_to_nearest_linestring(*linestring_quadrant_pairs,
                                                                 *quadtree,
                                                                 point_indices,
                                                                 x,
                                                                 y,
                                                                 *linestring_offsets,
                                                                 *linestring_x,
                                                                 *linestring_y),
                 cuspatial::logic_error);
  }
}
