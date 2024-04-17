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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/bounding_boxes.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/point_quadtree.cuh>
#include <cuspatial/range/multipolygon_range.cuh>
#include <cuspatial/spatial_join.cuh>

#include <gtest/gtest.h>

#include <initializer_list>

/*
 * A small test that it is suitable for manually visualizing point-polygon pairing results in a GIS
 * environment. GPU results are compared with expected values embedded in code. However, the number
 * of points in each quadrant is less than 32, the two kernels for point-in-polygon test are not
 * fully tested.
 */
template <typename T>
struct PIPRefineTestSmall : public cuspatial::test::BaseFixture {};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PIPRefineTestSmall, TestTypes);

TYPED_TEST(PIPRefineTestSmall, TestSmall)
{
  using T = TypeParam;
  using cuspatial::vec_2d;
  using cuspatial::test::make_device_vector;

  vec_2d<T> v_min{0.0, 0.0};
  vec_2d<T> v_max{8.0, 8.0};
  T const scale{1.0};
  uint8_t const max_depth{3};
  uint32_t const max_size{12};

  auto points = make_device_vector<vec_2d<T>>(
    {{1.9804558865545805, 1.3472225743317712},  {0.1895259128530169, 0.5431061133894604},
     {1.2591725716781235, 0.1448705855995005},  {0.8178039499335275, 0.8138440641113271},
     {0.48171647380517046, 1.9022922214961997}, {1.3890664414691907, 1.5177694304735412},
     {0.2536015260915061, 1.8762161698642947},  {3.1907684812039956, 0.2621847215928189},
     {3.028362149164369, 0.027638405909631958}, {3.918090468102582, 0.3338651960183463},
     {3.710910700915217, 0.9937713340192049},   {3.0706987088385853, 0.9376313558467103},
     {3.572744183805594, 0.33184908855075124},  {3.7080407833612004, 0.09804238103130436},
     {3.70669993057843, 0.7485845679979923},    {3.3588457228653024, 0.2346381514128677},
     {2.0697434332621234, 1.1809465376402173},  {2.5322042870739683, 1.419555755682142},
     {2.175448214220591, 1.2372448404986038},   {2.113652420701984, 1.2774712415624014},
     {2.520755151373394, 1.902015274420646},    {2.9909779614491687, 1.2420487904041893},
     {2.4613232527836137, 1.0484414482621331},  {4.975578758530645, 0.9606291981013242},
     {4.07037627210835, 1.9486902798139454},    {4.300706849071861, 0.021365525588281198},
     {4.5584381091040616, 1.8996548860019926},  {4.822583857757069, 0.3234041700489503},
     {4.849847745942472, 1.9531893897409585},   {4.75489831780737, 0.7800065259479418},
     {4.529792124514895, 1.942673409259531},    {4.732546857961497, 0.5659923375279095},
     {3.7622247877537456, 2.8709552313924487},  {3.2648444465931474, 2.693039435509084},
     {3.01954722322135, 2.57810040095543},      {3.7164018490892348, 2.4612194182614333},
     {3.7002781846945347, 2.3345952955903906},  {2.493975723955388, 3.3999020934055837},
     {2.1807636574967466, 3.2296461832828114},  {2.566986568683904, 3.6607732238530897},
     {2.2006520196663066, 3.7672478678985257},  {2.5104987015171574, 3.0668114607133137},
     {2.8222482218882474, 3.8159308233351266},  {2.241538022180476, 3.8812819070357545},
     {2.3007438625108882, 3.6045900851589048},  {6.0821276168848994, 2.5470532680258002},
     {6.291790729917634, 2.983311357415729},    {6.109985464455084, 2.2235950639628523},
     {6.101327777646798, 2.5239201807166616},   {6.325158445513714, 2.8765450351723674},
     {6.6793884701899, 2.5605928243991434},     {6.4274219368674315, 2.9754616970668213},
     {6.444584786789386, 2.174562817047202},    {7.897735998643542, 3.380784914178574},
     {7.079453687660189, 3.063690547962938},    {7.430677191305505, 3.380489849365283},
     {7.5085184104988, 3.623862886287816},      {7.886010001346151, 3.538128217886674},
     {7.250745898479374, 3.4154469467473447},   {7.769497359206111, 3.253257011908445},
     {1.8703303641352362, 4.209727933188015},   {1.7015273093278767, 7.478882372510933},
     {2.7456295127617385, 7.474216636277054},   {2.2065031771469, 6.896038613284851},
     {3.86008672302403, 7.513564222799629},     {1.9143371250907073, 6.885401350515916},
     {3.7176098065039747, 6.194330707468438},   {0.059011873032214, 5.823535317960799},
     {3.1162712022943757, 6.789029097334483},   {2.4264509160270813, 5.188939408363776},
     {3.154282922203257, 5.788316610960881}});

  // build a quadtree on the points
  auto [point_indices, quadtree] = cuspatial::quadtree_on_points(
    points.begin(), points.end(), v_min, v_max, scale, max_depth, max_size, this->stream());

  auto multipoly_array = cuspatial::test::make_multipolygon_array<T>({0, 1, 2, 3, 4},
                                                                     {0, 1, 2, 3, 4},
                                                                     {0, 4, 10, 14, 19},
                                                                     {// ring 1
                                                                      {2.488450, 5.856625},
                                                                      {1.333584, 5.008840},
                                                                      {3.460720, 4.586599},
                                                                      {2.488450, 5.856625},
                                                                      // ring 2
                                                                      {5.039823, 4.229242},
                                                                      {5.561707, 1.825073},
                                                                      {7.103516, 1.503906},
                                                                      {7.190674, 4.025879},
                                                                      {5.998939, 5.653384},
                                                                      {5.039823, 4.229242},
                                                                      // ring 3
                                                                      {5.998939, 1.235638},
                                                                      {5.573720, 0.197808},
                                                                      {6.703534, 0.086693},
                                                                      {5.998939, 1.235638},
                                                                      // ring 4
                                                                      {2.088115, 4.541529},
                                                                      {1.034892, 3.530299},
                                                                      {2.415080, 2.896937},
                                                                      {3.208660, 3.745936},
                                                                      {2.088115, 4.541529}});
  auto multipolygons   = multipoly_array.range();

  auto bboxes =
    rmm::device_uvector<cuspatial::box<T>>(multipolygons.num_polygons(), this->stream());

  cuspatial::polygon_bounding_boxes(multipolygons.part_offset_begin(),
                                    multipolygons.part_offset_end(),
                                    multipolygons.ring_offset_begin(),
                                    multipolygons.ring_offset_end(),
                                    multipolygons.point_begin(),
                                    multipolygons.point_end(),
                                    bboxes.begin(),
                                    T{0},
                                    this->stream());

  auto [poly_indices, quad_indices] = cuspatial::join_quadtree_and_bounding_boxes(
    quadtree, bboxes.begin(), bboxes.end(), v_min, scale, max_depth, this->stream(), this->mr());

  {
    auto expected_poly_indices = make_device_vector<uint32_t>({3, 3, 1, 2, 1, 1, 0, 3});
    auto expected_quad_indices = make_device_vector<uint32_t>({10, 11, 6, 6, 12, 13, 2, 2});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(poly_indices, expected_poly_indices);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(quad_indices, expected_quad_indices);
  }

  auto [poly_offset, point_offset] = cuspatial::quadtree_point_in_polygon(poly_indices.begin(),
                                                                          poly_indices.end(),
                                                                          quad_indices.begin(),
                                                                          quadtree,
                                                                          point_indices.begin(),
                                                                          point_indices.end(),
                                                                          points.begin(),
                                                                          multipolygons,
                                                                          this->stream());

  auto expected_poly_offset =
    make_device_vector<uint32_t>({3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3});
  auto expected_point_offset = make_device_vector<uint32_t>(
    {28, 29, 30, 31, 32, 33, 34, 35, 45, 46, 47, 48, 49, 50, 51, 52, 54, 62, 60});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(poly_offset, expected_poly_offset);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(point_offset, expected_point_offset);
}
