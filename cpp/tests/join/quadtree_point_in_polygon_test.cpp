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

struct QuadtreePointInPolygonErrorTest : public ::testing::Test {
  auto prepare_test(cudf::column_view const& x,
                    cudf::column_view const& y,
                    cudf::column_view const& polygon_offsets,
                    cudf::column_view const& ring_offsets,
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

    auto polygon_bboxes = cuspatial::polygon_bounding_boxes(
      polygon_offsets, ring_offsets, polygon_x, polygon_y, expansion_radius);

    auto polygon_quadrant_pairs = cuspatial::join_quadtree_and_bounding_boxes(
      *quadtree, *polygon_bboxes, v_min.x, v_max.x, v_min.y, v_max.y, scale, max_depth);

    return std::make_tuple(std::move(quadtree),
                           std::move(point_indices),
                           std::move(polygon_bboxes),
                           std::move(polygon_quadrant_pairs));
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

    auto polygon_offsets_col = wrapper<std::int32_t>({0, 4, 10});
    auto ring_offsets_col    = wrapper<std::int32_t>({0, 4, 10});
    auto polygon_x_col       = wrapper<T>({// ring 1
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
    auto polygon_y_col       = wrapper<T>({// ring 1
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

    std::tie(quadtree, point_indices, polygon_bboxes, polygon_quadrant_pairs) =
      prepare_test(x_col,
                   y_col,
                   polygon_offsets_col,
                   ring_offsets_col,
                   polygon_x_col,
                   polygon_y_col,
                   v_min,
                   v_max,
                   scale,
                   max_depth,
                   min_size,
                   expansion_radius);

    x               = x_col.release();
    y               = y_col.release();
    polygon_offsets = polygon_offsets_col.release();
    ring_offsets    = ring_offsets_col.release();
    polygon_x       = polygon_x_col.release();
    polygon_y       = polygon_y_col.release();
  }

  void TearDown() override {}

  std::unique_ptr<cudf::column> x;
  std::unique_ptr<cudf::column> y;
  std::unique_ptr<cudf::column> polygon_offsets;
  std::unique_ptr<cudf::column> ring_offsets;
  std::unique_ptr<cudf::column> polygon_x;
  std::unique_ptr<cudf::column> polygon_y;
  std::unique_ptr<cudf::column> point_indices;
  std::unique_ptr<cudf::table> quadtree;
  std::unique_ptr<cudf::table> polygon_bboxes;
  std::unique_ptr<cudf::table> polygon_quadrant_pairs;
};

// test cudf::quadtree_point_in_polygon with empty inputs
TEST_F(QuadtreePointInPolygonErrorTest, test_empty)
{
  // empty point data
  {
    auto empty_point_indices = wrapper<std::uint32_t>({});
    auto empty_x             = wrapper<T>({});
    auto empty_y             = wrapper<T>({});

    auto results = cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                        *quadtree,
                                                        empty_point_indices,
                                                        empty_x,
                                                        empty_y,
                                                        *polygon_offsets,
                                                        *ring_offsets,
                                                        *polygon_x,
                                                        *polygon_y);

    auto expected_poly_offset  = wrapper<std::uint32_t>({});
    auto expected_point_offset = wrapper<std::uint32_t>({});

    auto expected = cudf::table_view{
      {cudf::column_view(expected_poly_offset), cudf::column_view(expected_point_offset)}};

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *results);
  }

  // empty polygon data
  {
    auto empty_polygon_offsets = wrapper<std::uint32_t>({});
    auto empty_ring_offsets    = wrapper<std::uint32_t>({});
    auto empty_polygon_x       = wrapper<T>({});
    auto empty_polygon_y       = wrapper<T>({});

    auto results = cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                        *quadtree,
                                                        *point_indices,
                                                        *x,
                                                        *y,
                                                        empty_polygon_offsets,
                                                        empty_ring_offsets,
                                                        empty_polygon_x,
                                                        empty_polygon_y);

    auto expected_poly_offset  = wrapper<std::uint32_t>({});
    auto expected_point_offset = wrapper<std::uint32_t>({});

    auto expected = cudf::table_view{
      {cudf::column_view(expected_poly_offset), cudf::column_view(expected_point_offset)}};

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *results);
  }
}

TEST_F(QuadtreePointInPolygonErrorTest, type_mismatch)
{
  // x/y type mismatch
  {
    auto x_col = wrapper<int32_t>({1, 2, 3, 4});
    auto y_col = wrapper<float>({1, 2, 3, 4});

    EXPECT_THROW(cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                      *quadtree,
                                                      *point_indices,
                                                      x_col,
                                                      y_col,
                                                      *polygon_offsets,
                                                      *ring_offsets,
                                                      *polygon_x,
                                                      *polygon_y),
                 cuspatial::logic_error);
  }

  // polygon_x/polygon_y type mismatch
  {
    auto polygon_x_col = wrapper<int32_t>({1, 2, 3, 4});
    auto polygon_y_col = wrapper<float>({1, 2, 3, 4});

    EXPECT_THROW(cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                      *quadtree,
                                                      *point_indices,
                                                      *x,
                                                      *y,
                                                      *polygon_offsets,
                                                      *ring_offsets,
                                                      polygon_x_col,
                                                      polygon_y_col),
                 cuspatial::logic_error);
  }

  // x / polygon_x type mismatch
  {
    auto x_col         = wrapper<int32_t>({1, 2, 3, 4});
    auto polygon_x_col = wrapper<float>({1, 2, 3, 4});

    EXPECT_THROW(cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                      *quadtree,
                                                      *point_indices,
                                                      x_col,
                                                      *y,
                                                      *polygon_offsets,
                                                      *ring_offsets,
                                                      polygon_x_col,
                                                      *polygon_y),
                 cuspatial::logic_error);
  }
}

TEST_F(QuadtreePointInPolygonErrorTest, offset_type_error)
{
  {
    auto polygon_offsets_col = wrapper<float>({0, 4, 10});
    auto ring_offsets_col    = wrapper<std::int32_t>({0, 4, 10});

    EXPECT_THROW(cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                      *quadtree,
                                                      *point_indices,
                                                      *x,
                                                      *y,
                                                      polygon_offsets_col,
                                                      ring_offsets_col,
                                                      *polygon_x,
                                                      *polygon_y),
                 cuspatial::logic_error);
  }
}

TEST_F(QuadtreePointInPolygonErrorTest, size_mismatch)
{
  {
    auto polygon_offsets_col = wrapper<int32_t>({0, 4, 10});
    auto ring_offsets_col    = wrapper<int32_t>({0, 4, 10});
    auto poly_x              = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto poly_y              = wrapper<float>({1, 2, 3, 4, 5, 6});

    EXPECT_THROW(cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                      *quadtree,
                                                      *point_indices,
                                                      *x,
                                                      *y,
                                                      polygon_offsets_col,
                                                      ring_offsets_col,
                                                      poly_x,
                                                      poly_y),
                 cuspatial::logic_error);
  }
  {
    auto point_indices = wrapper<int32_t>({0, 1, 2, 3, 4});
    auto x             = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto y             = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    EXPECT_THROW(cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                      *quadtree,
                                                      point_indices,
                                                      x,
                                                      y,
                                                      *polygon_offsets,
                                                      *ring_offsets,
                                                      *polygon_x,
                                                      *polygon_y),
                 cuspatial::logic_error);
  }
  {
    auto point_indices = wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto x             = wrapper<float>({1, 2, 3, 4, 5});
    auto y             = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    EXPECT_THROW(cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                      *quadtree,
                                                      point_indices,
                                                      x,
                                                      y,
                                                      *polygon_offsets,
                                                      *ring_offsets,
                                                      *polygon_x,
                                                      *polygon_y),
                 cuspatial::logic_error);
  }
  {
    auto point_indices = wrapper<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto x             = wrapper<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto y             = wrapper<float>({1, 2, 3, 4, 5});

    EXPECT_THROW(cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                      *quadtree,
                                                      point_indices,
                                                      x,
                                                      y,
                                                      *polygon_offsets,
                                                      *ring_offsets,
                                                      *polygon_x,
                                                      *polygon_y),
                 cuspatial::logic_error);
  }
}
