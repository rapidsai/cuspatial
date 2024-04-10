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

#include <cuspatial/error.hpp>
#include <cuspatial/point_quadtree.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

struct QuadtreeOnPointErrorTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(QuadtreeOnPointIndexingTest, cudf::test::FloatingPointTypes);

TEST_F(QuadtreeOnPointErrorTest, test_empty)
{
  using T = float;
  using namespace cudf::test;
  const int8_t max_depth = 1;
  uint32_t min_size      = 1;
  double scale           = 1.0;
  double x_min = 0, x_max = 1, y_min = 0, y_max = 1;

  fixed_width_column_wrapper<T> x({});
  fixed_width_column_wrapper<T> y({});

  auto quadtree_pair =
    cuspatial::quadtree_on_points(x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size);
  auto& quadtree = std::get<1>(quadtree_pair);

  CUSPATIAL_EXPECTS(
    quadtree->num_columns() == 5,
    "a quadtree table must have 5 columns (keys, levels, is_node, lengths, offsets)");

  CUSPATIAL_EXPECTS(quadtree->num_rows() == 0,
                    "the resulting quadtree must have a single quadrant");
}

TEST_F(QuadtreeOnPointErrorTest, test_x_y_size_mismatch)
{
  using T = float;
  using namespace cudf::test;
  const int8_t max_depth = 1;
  uint32_t min_size      = 1;
  double scale           = 1.0;
  double x_min = 0, x_max = 1, y_min = 0, y_max = 1;

  fixed_width_column_wrapper<T> x({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  fixed_width_column_wrapper<T> y({0, 1, 2, 3, 4, 5, 6, 7, 8});

  EXPECT_THROW(
    cuspatial::quadtree_on_points(x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size),
    cuspatial::logic_error);
}
