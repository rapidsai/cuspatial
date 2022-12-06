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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

template <typename T>
struct QuadtreeOnPointIndexingTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(QuadtreeOnPointIndexingTest, cudf::test::FloatingPointTypes);

TYPED_TEST(QuadtreeOnPointIndexingTest, test_empty)
{
  using T = TypeParam;
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

TYPED_TEST(QuadtreeOnPointIndexingTest, test_single)
{
  using T = TypeParam;
  using namespace cudf::test;
  const int8_t max_depth = 1;
  uint32_t min_size      = 1;

  double scale = 1.0;
  double x_min = 0, x_max = 1, y_min = 0, y_max = 1;

  fixed_width_column_wrapper<T> x({0.45});
  fixed_width_column_wrapper<T> y({0.45});

  auto quadtree_pair =
    cuspatial::quadtree_on_points(x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size);
  auto& quadtree = std::get<1>(quadtree_pair);

  CUSPATIAL_EXPECTS(
    quadtree->num_columns() == 5,
    "a quadtree table must have 5 columns (keys, levels, is_node, lengths, offsets)");

  CUSPATIAL_EXPECTS(quadtree->num_rows() == 1,
                    "the resulting quadtree must have a single quadrant");

  // the top level quadtree node is expected to have a value of (0,0,0,1,0)
  fixed_width_column_wrapper<uint32_t> expected1({0});
  fixed_width_column_wrapper<uint8_t> expected2({0});
  fixed_width_column_wrapper<bool> expected3({0});
  fixed_width_column_wrapper<uint32_t> expected4({1});
  fixed_width_column_wrapper<uint32_t> expected5({0});
  auto expected = cudf::table_view{{expected1, expected2, expected3, expected4, expected5}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*quadtree, expected);
}

TYPED_TEST(QuadtreeOnPointIndexingTest, test_two)
{
  using T = TypeParam;
  using namespace cudf::test;

  const int8_t max_depth = 1;
  uint32_t min_size      = 1;

  double scale = 1.0;
  double x_min = 0, x_max = 2, y_min = 0, y_max = 2;

  fixed_width_column_wrapper<T> x({0.45, 1.45});
  fixed_width_column_wrapper<T> y({0.45, 1.45});

  auto quadtree_pair =
    cuspatial::quadtree_on_points(x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size);
  auto& quadtree = std::get<1>(quadtree_pair);

  CUSPATIAL_EXPECTS(
    quadtree->num_columns() == 5,
    "a quadtree table must have 5 columns (keys, levels, is_node, lengths, offsets)");

  CUSPATIAL_EXPECTS(quadtree->num_rows() == 2, "the resulting quadtree must have 2 quadrants");

  // the top level quadtree node is expected to have a value of
  // ([0, 3], [0, 0], [0, 0], [1, 1], [0, 1])
  fixed_width_column_wrapper<uint32_t> expected1({0, 3});
  fixed_width_column_wrapper<uint8_t> expected2({0, 0});
  fixed_width_column_wrapper<bool> expected3({0, 0});
  fixed_width_column_wrapper<uint32_t> expected4({1, 1});
  fixed_width_column_wrapper<uint32_t> expected5({0, 1});

  auto expected = cudf::table_view{{expected1, expected2, expected3, expected4, expected5}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(*quadtree, expected);
}

TYPED_TEST(QuadtreeOnPointIndexingTest, test_small)
{
  using T = TypeParam;
  using namespace cudf::test;

  const int8_t max_depth = 3;
  uint32_t min_size      = 12;
  double scale           = 1.0;
  double x_min = 0, x_max = 8, y_min = 0, y_max = 8;

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

  auto quadtree_pair =
    cuspatial::quadtree_on_points(x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size);
  auto& quadtree = std::get<1>(quadtree_pair);

  CUSPATIAL_EXPECTS(
    quadtree->num_columns() == 5,
    "a quadtree table must have 5 columns (keys, levels, is_node, lengths, offsets)");

  fixed_width_column_wrapper<uint32_t> expected1({0, 1, 2, 0, 1, 3, 4, 7, 5, 6, 13, 14, 28, 31});
  fixed_width_column_wrapper<uint8_t> expected2({0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2});
  fixed_width_column_wrapper<bool> expected3({1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0});
  fixed_width_column_wrapper<uint32_t> expected4({3, 2, 11, 7, 2, 2, 9, 2, 9, 7, 5, 8, 8, 7});
  fixed_width_column_wrapper<uint32_t> expected5(
    {3, 6, 60, 0, 8, 10, 36, 12, 7, 16, 23, 28, 45, 53});

  auto expected = cudf::table_view{{expected1, expected2, expected3, expected4, expected5}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(*quadtree, expected);
}

TYPED_TEST(QuadtreeOnPointIndexingTest, test_all_lowest_level_quads)
{
  using T = TypeParam;
  using namespace cudf::test;

  const int8_t max_depth = 2;
  uint32_t min_size      = 1;

  double x_min = -1000.0;
  double x_max = 1000.0;
  double y_min = -1000.0;
  double y_max = 1000.0;
  double scale = std::max(x_max - x_min, y_max - y_min) / static_cast<double>((1 << max_depth) + 2);

  fixed_width_column_wrapper<T> x({-100.0, 100.0});
  fixed_width_column_wrapper<T> y({-100.0, 100.0});

  auto quadtree_pair =
    cuspatial::quadtree_on_points(x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size);
  auto& quadtree = std::get<1>(quadtree_pair);

  CUSPATIAL_EXPECTS(
    quadtree->num_columns() == 5,
    "a quadtree table must have 5 columns (keys, levels, is_node, lengths, offsets)");

  CUSPATIAL_EXPECTS(quadtree->num_rows() == 3, "the resulting quadtree must have 3 quadrants");

  // the top level quadtree node is expected to have a value of
  // ([3, 12, 15], [0, 1, 1], [1, 0, 0], [2, 1, 1], [1, 0, 1])

  fixed_width_column_wrapper<uint32_t> expected1({3, 12, 15});
  fixed_width_column_wrapper<uint8_t> expected2({0, 1, 1});
  fixed_width_column_wrapper<bool> expected3({1, 0, 0});
  fixed_width_column_wrapper<uint32_t> expected4({2, 1, 1});
  fixed_width_column_wrapper<uint32_t> expected5({1, 0, 1});

  auto expected = cudf::table_view{{expected1, expected2, expected3, expected4, expected5}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(*quadtree, expected);
}
