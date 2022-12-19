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
#include <cuspatial/polygon_bounding_box.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

struct PolygonBoundingBoxErrorTest : public ::testing::Test {
};

using T = float;

TEST_F(PolygonBoundingBoxErrorTest, test_empty)
{
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({});
  fixed_width_column_wrapper<int32_t> ring_offsets({});
  fixed_width_column_wrapper<T> x({});
  fixed_width_column_wrapper<T> y({});

  auto bboxes = cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0);

  EXPECT_EQ(bboxes->num_rows(), 0);
}

TEST_F(PolygonBoundingBoxErrorTest, test_more_polys_than_rings)
{
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({0, 1, 2});
  fixed_width_column_wrapper<int32_t> ring_offsets({0, 4});
  fixed_width_column_wrapper<T> x({2.488450, 1.333584, 3.460720, 2.488450});
  fixed_width_column_wrapper<T> y({5.856625, 5.008840, 4.586599, 5.856625});

  EXPECT_THROW(cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0),
               cuspatial::logic_error);
}

TEST_F(PolygonBoundingBoxErrorTest, type_mismatch)
{
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({0, 1});
  fixed_width_column_wrapper<int32_t> ring_offsets({0, 4});
  fixed_width_column_wrapper<T> x({2.488450, 1.333584, 3.460720, 2.488450});
  fixed_width_column_wrapper<double> y({5.856625, 5.008840, 4.586599, 5.856625});

  EXPECT_THROW(cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0),
               cuspatial::logic_error);
}

TEST_F(PolygonBoundingBoxErrorTest, not_enough_offsets)
{
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({0});
  fixed_width_column_wrapper<int32_t> ring_offsets({0});
  fixed_width_column_wrapper<T> x({2.488450, 1.333584, 3.460720, 2.488450});
  fixed_width_column_wrapper<T> y({5.856625, 5.008840, 4.586599, 5.856625});

  auto bboxes = cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0);

  EXPECT_EQ(bboxes->num_rows(), 0);
}

TEST_F(PolygonBoundingBoxErrorTest, offset_type_error)
{
  using namespace cudf::test;

  {
    fixed_width_column_wrapper<float> poly_offsets({0, 1});
    fixed_width_column_wrapper<int32_t> ring_offsets({0, 4});
    fixed_width_column_wrapper<T> x({2.488450, 1.333584, 3.460720, 2.488450});
    fixed_width_column_wrapper<T> y({5.856625, 5.008840, 4.586599, 5.856625});

    EXPECT_THROW(cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0),
                 cuspatial::logic_error);
  }
  {
    fixed_width_column_wrapper<int32_t> poly_offsets({0, 1});
    fixed_width_column_wrapper<float> ring_offsets({0, 4});
    fixed_width_column_wrapper<T> x({2.488450, 1.333584, 3.460720, 2.488450});
    fixed_width_column_wrapper<T> y({5.856625, 5.008840, 4.586599, 5.856625});

    EXPECT_THROW(cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0),
                 cuspatial::logic_error);
  }
}

TEST_F(PolygonBoundingBoxErrorTest, vertex_size_mismatch)
{
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({0, 1});
  fixed_width_column_wrapper<int32_t> ring_offsets({0, 4});
  fixed_width_column_wrapper<T> x({2.488450, 1.333584, 3.460720, 2.488450});
  fixed_width_column_wrapper<T> y({5.856625, 5.008840});

  EXPECT_THROW(cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0),
               cuspatial::logic_error);
}

TEST_F(PolygonBoundingBoxErrorTest, ring_too_small)
{
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({0, 1});
  fixed_width_column_wrapper<int32_t> ring_offsets({0, 2});
  fixed_width_column_wrapper<T> x({2.488450, 1.333584});
  fixed_width_column_wrapper<T> y({5.856625, 5.008840});

  EXPECT_THROW(cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0),
               cuspatial::logic_error);
}
