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
#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/polygon_bounding_box.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

template <typename T>
struct QuadtreePolygonFilteringTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(QuadtreePolygonFilteringTest, cudf::test::FloatingPointTypes);

TYPED_TEST(QuadtreePolygonFilteringTest, test_errors)
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

TYPED_TEST(QuadtreePolygonFilteringTest, test_empty)
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

TYPED_TEST(QuadtreePolygonFilteringTest, test_small)
{
  using T = TypeParam;
  using namespace cudf::test;

  double const x_min{0.0};
  double const x_max{8.0};
  double const y_min{0.0};
  double const y_max{8.0};
  double const scale{1.0};
  uint32_t const max_depth{3};
  uint32_t const min_size{12};

  fixed_width_column_wrapper<T> x(
    {1.9804558865545805,  0.1895259128530169, 1.2591725716781235, 0.8178039499335275,
     0.48171647380517046, 1.3890664414691907, 0.2536015260915061, 3.1907684812039956,
     3.028362149164369,   3.918090468102582,  3.710910700915217,  3.0706987088385853,
     3.572744183805594,   3.7080407833612004, 3.70669993057843,   3.3588457228653024,
     2.0697434332621234,  2.5322042870739683, 2.175448214220591,  2.113652420701984,
     2.520755151373394,   2.9909779614491687, 2.4613232527836137, 4.975578758530645,
     4.07037627210835,    4.300706849071861,  4.5584381091040616, 4.822583857757069,
     4.849847745942472,   4.75489831780737,   4.529792124514895,  4.732546857961497,
     3.7622247877537456,  3.2648444465931474, 3.01954722322135,   3.7164018490892348,
     3.7002781846945347,  2.493975723955388,  2.1807636574967466, 2.566986568683904,
     2.2006520196663066,  2.5104987015171574, 2.8222482218882474, 2.241538022180476,
     2.3007438625108882,  6.0821276168848994, 6.291790729917634,  6.109985464455084,
     6.101327777646798,   6.325158445513714,  6.6793884701899,    6.4274219368674315,
     6.444584786789386,   7.897735998643542,  7.079453687660189,  7.430677191305505,
     7.5085184104988,     7.886010001346151,  7.250745898479374,  7.769497359206111,
     1.8703303641352362,  1.7015273093278767, 2.7456295127617385, 2.2065031771469,
     3.86008672302403,    1.9143371250907073, 3.7176098065039747, 0.059011873032214,
     3.1162712022943757,  2.4264509160270813, 3.154282922203257});

  fixed_width_column_wrapper<T> y(
    {1.3472225743317712,   0.5431061133894604,   0.1448705855995005, 0.8138440641113271,
     1.9022922214961997,   1.5177694304735412,   1.8762161698642947, 0.2621847215928189,
     0.027638405909631958, 0.3338651960183463,   0.9937713340192049, 0.9376313558467103,
     0.33184908855075124,  0.09804238103130436,  0.7485845679979923, 0.2346381514128677,
     1.1809465376402173,   1.419555755682142,    1.2372448404986038, 1.2774712415624014,
     1.902015274420646,    1.2420487904041893,   1.0484414482621331, 0.9606291981013242,
     1.9486902798139454,   0.021365525588281198, 1.8996548860019926, 0.3234041700489503,
     1.9531893897409585,   0.7800065259479418,   1.942673409259531,  0.5659923375279095,
     2.8709552313924487,   2.693039435509084,    2.57810040095543,   2.4612194182614333,
     2.3345952955903906,   3.3999020934055837,   3.2296461832828114, 3.6607732238530897,
     3.7672478678985257,   3.0668114607133137,   3.8159308233351266, 3.8812819070357545,
     3.6045900851589048,   2.5470532680258002,   2.983311357415729,  2.2235950639628523,
     2.5239201807166616,   2.8765450351723674,   2.5605928243991434, 2.9754616970668213,
     2.174562817047202,    3.380784914178574,    3.063690547962938,  3.380489849365283,
     3.623862886287816,    3.538128217886674,    3.4154469467473447, 3.253257011908445,
     4.209727933188015,    7.478882372510933,    7.474216636277054,  6.896038613284851,
     7.513564222799629,    6.885401350515916,    6.194330707468438,  5.823535317960799,
     6.789029097334483,    5.188939408363776,    5.788316610960881});

  auto pair = cuspatial::quadtree_on_points(
    x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size, this->mr());

  auto& quadtree = std::get<1>(pair);

  fixed_width_column_wrapper<int32_t> poly_offsets({0, 1, 2, 3, 4});
  fixed_width_column_wrapper<int32_t> ring_offsets({0, 3, 8, 12, 17});
  fixed_width_column_wrapper<T> poly_x({// ring 1
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
  fixed_width_column_wrapper<T> poly_y({// ring 1
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

  auto polygon_bboxes =
    cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, poly_x, poly_y, 0.0, this->mr());

  auto polygon_quadrant_pairs = cuspatial::join_quadtree_and_bounding_boxes(
    *quadtree, *polygon_bboxes, x_min, x_max, y_min, y_max, scale, max_depth, this->mr());

  CUSPATIAL_EXPECTS(
    polygon_quadrant_pairs->num_columns() == 2,
    "a polygon-quadrant pair table must have 2 columns (polygon-index, quadrant-index)");

  fixed_width_column_wrapper<uint32_t> expected1({3, 3, 1, 2, 1, 1, 0, 3});
  fixed_width_column_wrapper<uint32_t> expected2({10, 11, 6, 6, 12, 13, 2, 2});
  auto expected = cudf::table_view{{expected1, expected2}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *polygon_quadrant_pairs);
}
