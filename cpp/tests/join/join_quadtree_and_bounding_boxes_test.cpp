/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

template <typename T>
struct JoinQuadtreeAndBoundingBoxesErrorTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(JoinQuadtreeAndBoundingBoxesErrorTest, cudf::test::FloatingPointTypes);

TYPED_TEST(JoinQuadtreeAndBoundingBoxesErrorTest, test_errors)
{
  using T = TypeParam;
  using namespace cudf::test;
  // bad table
  cudf::table_view bad_quadtree{};
  // bad bboxes
  cudf::table_view bad_bboxes{};
  // empty quadtree
  cudf::table_view empty_quadtree{{
    fixed_width_column_wrapper<int32_t>({}),
    fixed_width_column_wrapper<int8_t>({}),
    fixed_width_column_wrapper<bool>({}),
    fixed_width_column_wrapper<int32_t>({}),
    fixed_width_column_wrapper<int32_t>({}),
  }};
  // empty bboxes
  cudf::table_view empty_bboxes{{fixed_width_column_wrapper<T>({}),
                                 fixed_width_column_wrapper<T>({}),
                                 fixed_width_column_wrapper<T>({}),
                                 fixed_width_column_wrapper<T>({})}};

  // Test throws on bad quadtree
  EXPECT_THROW(cuspatial::join_quadtree_and_bounding_boxes(
                 bad_quadtree, empty_bboxes, 0, 1, 0, 1, 1, 1, this->mr()),
               cuspatial::logic_error);

  // Test throws on bad bboxes
  EXPECT_THROW(cuspatial::join_quadtree_and_bounding_boxes(
                 empty_quadtree, bad_bboxes, 0, 1, 0, 1, 1, 1, this->mr()),
               cuspatial::logic_error);

  // Test throws on bad scale
  EXPECT_THROW(cuspatial::join_quadtree_and_bounding_boxes(
                 empty_quadtree, empty_bboxes, 0, 1, 0, 1, 0, 1, this->mr()),
               cuspatial::logic_error);

  // Test throws on bad max_depth <= 0
  EXPECT_THROW(cuspatial::join_quadtree_and_bounding_boxes(
                 empty_quadtree, empty_bboxes, 0, 1, 0, 1, 1, 0, this->mr()),
               cuspatial::logic_error);

  // Test throws on bad max_depth >= 16
  EXPECT_THROW(cuspatial::join_quadtree_and_bounding_boxes(
                 empty_quadtree, empty_bboxes, 0, 1, 0, 1, 1, 16, this->mr()),
               cuspatial::logic_error);

  // Test throws on reversed area of interest bbox coordinates
  EXPECT_THROW(cuspatial::join_quadtree_and_bounding_boxes(
                 empty_quadtree, empty_bboxes, 1, 0, 1, 0, 1, 1, this->mr()),
               cuspatial::logic_error);
}

TYPED_TEST(JoinQuadtreeAndBoundingBoxesErrorTest, test_empty)
{
  using T = TypeParam;
  using namespace cudf::test;

  double const x_min{0.0};
  double const x_max{1.0};
  double const y_min{0.0};
  double const y_max{1.0};
  double const scale{1.0};
  uint32_t const max_depth{1};

  // empty quadtree
  cudf::table_view quadtree{{
    fixed_width_column_wrapper<int32_t>({}),
    fixed_width_column_wrapper<int8_t>({}),
    fixed_width_column_wrapper<bool>({}),
    fixed_width_column_wrapper<int32_t>({}),
    fixed_width_column_wrapper<int32_t>({}),
  }};
  // empty bboxes
  cudf::table_view bboxes{{fixed_width_column_wrapper<T>({}),
                           fixed_width_column_wrapper<T>({}),
                           fixed_width_column_wrapper<T>({}),
                           fixed_width_column_wrapper<T>({})}};

  auto polygon_quadrant_pairs = cuspatial::join_quadtree_and_bounding_boxes(
    quadtree, bboxes, x_min, x_max, y_min, y_max, scale, max_depth, this->mr());

  auto expect_first  = fixed_width_column_wrapper<uint32_t>({});
  auto expect_second = fixed_width_column_wrapper<uint32_t>({});
  auto expect        = cudf::table_view{{expect_first, expect_second}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect, *polygon_quadrant_pairs);
}
