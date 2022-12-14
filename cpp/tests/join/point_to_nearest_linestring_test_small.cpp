/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cuspatial/linestring_bounding_box.hpp>
#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <type_traits>

/*
 * A small test that it is suitable for manually visualizing point-polygon pairing results in a GIS
 * environment GPU results are compared with expected values embeded in code However, the number of
 * points in each quadrant is less than 32, the two kernels for point-in-polygon test are not fully
 * tested. This is left for pip_refine_test_large.
 */
template <typename T>
struct PIPRefineTestSmall : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(PIPRefineTestSmall, cudf::test::FloatingPointTypes);

TYPED_TEST(PIPRefineTestSmall, TestSmall)
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

  auto quadtree_pair = cuspatial::quadtree_on_points(
    x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size, this->mr());

  auto& quadtree      = std::get<1>(quadtree_pair);
  auto& point_indices = std::get<0>(quadtree_pair);

  double const expansion_radius{2.0};
  fixed_width_column_wrapper<int32_t> linestring_offsets({0, 4, 10, 14, 19});
  fixed_width_column_wrapper<T> linestring_x({// ring 1
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
                                              5.039823,
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
  fixed_width_column_wrapper<T> linestring_y({// ring 1
                                              5.856625,
                                              5.008840,
                                              4.586599,
                                              5.856625,
                                              // ring 2
                                              4.229242,
                                              1.825073,
                                              1.503906,
                                              4.025879,
                                              5.653384,
                                              4.229242,
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

  auto linestring_bboxes = cuspatial::linestring_bounding_boxes(
    linestring_offsets, linestring_x, linestring_y, expansion_radius, this->mr());

  auto polygon_quadrant_pairs = cuspatial::join_quadtree_and_bounding_boxes(
    *quadtree, *linestring_bboxes, x_min, x_max, y_min, y_max, scale, max_depth, this->mr());

  auto point_to_linestring_distances =
    cuspatial::quadtree_point_to_nearest_linestring(*polygon_quadrant_pairs,
                                                    *quadtree,
                                                    *point_indices,
                                                    x,
                                                    y,
                                                    linestring_offsets,
                                                    linestring_x,
                                                    linestring_y,
                                                    this->mr());

  CUSPATIAL_EXPECTS(point_to_linestring_distances->num_columns() == 3,
                    "a point-to-linestring distance table must have 3 columns");

  CUSPATIAL_EXPECTS(
    point_to_linestring_distances->num_rows() == point_indices->size(),
    "number of point-to-linestring distance pairs should be the same as number of points");

  auto expected_distances_column = []() {
    if (std::is_same<T, float>()) {
      return fixed_width_column_wrapper<T>(
        {3.06755614,   2.55945015,  2.98496079,   1.71036518,  1.82931805,   1.60950696,
         1.68141198,   2.38382101,  2.55103993,   1.66121042,  2.02551198,   2.06608653,
         2.0054605,    1.86834478,  1.94656599,   2.2151804,   1.75039434,   1.48201656,
         1.67690217,   1.6472789,   1.00051796,   1.75223088,  1.84907377,   1.00189602,
         0.760027468,  0.65931344,  1.24821293,   1.32290053,  0.285818338,  0.204662085,
         0.41061914,   0.566183507, 0.0462928228, 0.166630849, 0.449532568,  0.566757083,
         0.842694938,  1.2851826,   0.761564255,  0.978420198, 0.917963803,  1.43116546,
         0.964613676,  0.668479323, 0.983481824,  0.661732435, 0.862337708,  0.50195682,
         0.675588429,  0.825302362, 0.460371286,  0.726516545, 0.5221892,    0.728920817,
         0.0779202655, 0.262149751, 0.331539005,  0.711767673, 0.0811179057, 0.605163872,
         0.0885084718, 1.51270044,  0.389437437,  0.487170845, 1.17812812,   1.8030436,
         1.07697463,   1.1812768,   1.12407148,   1.63790822,  2.15100765});
    }
    return fixed_width_column_wrapper<T>(
      {3.0675562686570932,   2.5594501016565698,  2.9849608928964071,   1.7103652150920774,
       1.8293181280383963,   1.6095070428899729,  1.681412227243898,    2.3838209461314879,
       2.5510398428020409,   1.6612106150272572,  2.0255119347250292,   2.0660867596957564,
       2.005460353737949,    1.8683447535522375,  1.9465658908648766,   2.215180472008103,
       1.7503944159063249,   1.4820166799617225,  1.6769023397521503,   1.6472789467219351,
       1.0005181046076022,   1.7522309916961678,  1.8490738879835735,   1.0018961233717569,
       0.76002760100291122,  0.65931355999132091, 1.2482129257770731,   1.3229005055827028,
       0.28581819228716798,  0.20466187296772376, 0.41061901127492934,  0.56618357460517321,
       0.046292709584059538, 0.16663093663041179, 0.44953247369220306,  0.56675685520587671,
       0.8426949387264755,   1.2851826443010033,  0.7615641155638555,   0.97842040913621187,
       0.91796378078050755,  1.4311654461101424,  0.96461369875795078,  0.66847988653443491,
       0.98348202146010699,  0.66173276971965733, 0.86233789031448094,  0.50195678903916696,
       0.6755886291567379,   0.82530249944765133, 0.46037120394920633,  0.72651648874084795,
       0.52218906793095576,  0.72892093000338909, 0.077921089704128393, 0.26215098141130333,
       0.33153993710577778,  0.71176747526132511, 0.081119666144327182, 0.60516346789266895,
       0.088508309264124049, 1.5127004224070386,  0.38943741327066272,  0.48717099143018805,
       1.1781283344854494,   1.8030436222567465,  1.0769747770485747,   1.181276832710481,
       1.1240715558969043,   1.6379084234284416,  2.1510078772519496});
  }();

  fixed_width_column_wrapper<uint32_t> expected1(
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
     24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70});

  fixed_width_column_wrapper<uint32_t> expected2(
    {3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  auto expected = cudf::table_view{
    {cudf::column_view(expected1), cudf::column_view(expected2), expected_distances_column}};

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, *point_to_linestring_distances);
}
