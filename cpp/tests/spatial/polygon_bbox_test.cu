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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

template <typename T>
struct PolygonBoundingBoxTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(PolygonBoundingBoxTest, cudf::test::FloatingPointTypes);

TYPED_TEST(PolygonBoundingBoxTest, test_empty)
{
  using T = TypeParam;
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({});
  fixed_width_column_wrapper<int32_t> ring_offsets({});
  fixed_width_column_wrapper<T> x({});
  fixed_width_column_wrapper<T> y({});

  auto bboxes =
    cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0, this->mr());

  CUSPATIAL_EXPECTS(bboxes->num_rows() == 0, "must return 0 bounding boxes on empty input");
}

TYPED_TEST(PolygonBoundingBoxTest, test_one)
{
  using T = TypeParam;
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({0});
  fixed_width_column_wrapper<int32_t> ring_offsets({0});
  fixed_width_column_wrapper<T> x({2.488450, 1.333584, 3.460720});
  fixed_width_column_wrapper<T> y({5.856625, 5.008840, 4.586599});

  auto bboxes =
    cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0, this->mr());

  CUSPATIAL_EXPECTS(bboxes->view().num_columns() == 4, "bbox table must have 4 columns");
  CUSPATIAL_EXPECTS(bboxes->num_rows() == 1,
                    "resutling #of bounding boxes must be the same as # of polygons");

  fixed_width_column_wrapper<T> expected1({1.333584});
  fixed_width_column_wrapper<T> expected2({4.586599});
  fixed_width_column_wrapper<T> expected3({3.460720});
  fixed_width_column_wrapper<T> expected4({5.856625});
  auto expected = cudf::table_view{{expected1, expected2, expected3, expected4}};

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*bboxes, expected);
}

TYPED_TEST(PolygonBoundingBoxTest, test_small)
{
  using T = TypeParam;
  using namespace cudf::test;

  fixed_width_column_wrapper<int32_t> poly_offsets({0, 1, 2, 3});
  fixed_width_column_wrapper<int32_t> ring_offsets({0, 3, 8, 12});
  fixed_width_column_wrapper<T> x({// ring 1
                                   2.488450,
                                   1.333584,
                                   3.460720,
                                   // ring 2
                                   5.039823,
                                   5.561707,
                                   7.103516,
                                   7.190674,
                                   5.998939,
                                   // ring 3
                                   5.998939,
                                   5.573720,
                                   6.703534,
                                   5.998939,
                                   // ring 4
                                   2.088115,
                                   1.034892,
                                   2.415080,
                                   3.208660,
                                   2.088115});
  fixed_width_column_wrapper<T> y({// ring 1
                                   5.856625,
                                   5.008840,
                                   4.586599,
                                   // ring 2
                                   4.229242,
                                   1.825073,
                                   1.503906,
                                   4.025879,
                                   5.653384,
                                   // ring 3
                                   1.235638,
                                   0.197808,
                                   0.086693,
                                   1.235638,
                                   // ring 4
                                   4.541529,
                                   3.530299,
                                   2.896937,
                                   3.745936,
                                   4.541529});

  auto bboxes =
    cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, x, y, 0.0, this->mr());

  CUSPATIAL_EXPECTS(bboxes->view().num_columns() == 4, "bbox table must have 4 columns");
  CUSPATIAL_EXPECTS(bboxes->num_rows() == 4,
                    "resutling #of bounding boxes must be the same as # of polygons");
  fixed_width_column_wrapper<T> expected1(
    {1.3335840000000001, 5.0398230000000002, 5.5737199999999998, 1.0348919999999999});
  fixed_width_column_wrapper<T> expected2(
    {4.5865989999999996, 1.503906, 0.086693000000000006, 2.8969369999999999});
  fixed_width_column_wrapper<T> expected3(
    {3.4607199999999998, 7.1906739999999996, 6.7035340000000003, 3.2086600000000001});
  fixed_width_column_wrapper<T> expected4(
    {5.8566250000000002, 5.653384, 1.235638, 4.5415289999999997});

  auto expected = cudf::table_view{{expected1, expected2, expected3, expected4}};

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*bboxes, expected);
}
