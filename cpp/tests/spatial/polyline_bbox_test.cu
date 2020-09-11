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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/polyline_bounding_box.hpp>

#include <cudf/utilities/test/base_fixture.hpp>
#include <cudf/utilities/test/column_utilities.hpp>
#include <cudf/utilities/test/column_wrapper.hpp>
#include <cudf/utilities/test/table_utilities.hpp>
#include <cudf/utilities/test/type_lists.hpp>

template <typename T>
struct BoundingBoxTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(BoundingBoxTest, cudf::test::FloatingPointTypes);

TYPED_TEST(BoundingBoxTest, test_empty)
{
  using T = TypeParam;
  using namespace cudf::test;

  double const expansion_radius{0};
  fixed_width_column_wrapper<int32_t> poly_offsets({});
  fixed_width_column_wrapper<T> x({});
  fixed_width_column_wrapper<T> y({});

  auto bboxes =
    cuspatial::polyline_bounding_boxes(poly_offsets, x, y, expansion_radius, this->mr());

  CUSPATIAL_EXPECTS(bboxes->num_rows() == 0, "must return 0 bounding boxes on empty input");
}

TYPED_TEST(BoundingBoxTest, test_one)
{
  using T = TypeParam;
  using namespace cudf::test;

  double const expansion_radius{0};
  fixed_width_column_wrapper<int32_t> poly_offsets({0});
  fixed_width_column_wrapper<T> x({2.488450, 1.333584, 3.460720});
  fixed_width_column_wrapper<T> y({5.856625, 5.008840, 4.586599});

  auto bboxes =
    cuspatial::polyline_bounding_boxes(poly_offsets, x, y, expansion_radius, this->mr());

  CUSPATIAL_EXPECTS(bboxes->view().num_columns() == 4, "bbox table must have 4 columns");
  CUSPATIAL_EXPECTS(bboxes->num_rows() == 1,
                    "resutling #of bounding boxes must be the same as # of polygons");

  expect_tables_equivalent(
    *bboxes,
    cudf::table_view{{fixed_width_column_wrapper<T>({1.333584 - expansion_radius}),
                      fixed_width_column_wrapper<T>({4.586599 - expansion_radius}),
                      fixed_width_column_wrapper<T>({3.460720 + expansion_radius}),
                      fixed_width_column_wrapper<T>({5.856625 + expansion_radius})}});
}

TYPED_TEST(BoundingBoxTest, test_small)
{
  using T = TypeParam;
  using namespace cudf::test;

  double const expansion_radius{0.5};
  fixed_width_column_wrapper<int32_t> poly_offsets({0, 3, 8, 12});
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
    cuspatial::polyline_bounding_boxes(poly_offsets, x, y, expansion_radius, this->mr());

  CUSPATIAL_EXPECTS(bboxes->view().num_columns() == 4, "bbox table must have 4 columns");
  CUSPATIAL_EXPECTS(bboxes->num_rows() == 4,
                    "resutling #of bounding boxes must be the same as # of polygons");

  expect_tables_equivalent(
    *bboxes,
    cudf::table_view{
      {fixed_width_column_wrapper<T>(
         {0.8335840000000001, 4.5398230000000002, 5.0737199999999998, 0.53489199999999992}),
       fixed_width_column_wrapper<T>(
         {4.0865989999999996, 1.003906, -0.41330699999999998, 2.3969369999999999}),
       fixed_width_column_wrapper<T>(
         {3.9607199999999998, 7.6906739999999996, 7.2035340000000003, 3.7086600000000001}),
       fixed_width_column_wrapper<T>(
         {6.3566250000000002, 6.153384, 1.735638, 5.0415289999999997})}});
}
