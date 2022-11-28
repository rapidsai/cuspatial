/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/experimental/point_quadtree.cuh>
#include <cuspatial/vec_2d.hpp>

template <typename T>
struct QuadtreeOnPointIndexingTest : public ::testing::Test {
  void test(std::vector<cuspatial::vec_2d<T>> const& points,
            const cuspatial::vec_2d<T> vertex_1,
            const cuspatial::vec_2d<T> vertex_2,
            const T scale,
            const int8_t max_depth,
            const int32_t max_size,
            std::vector<uint32_t> const& expected_key,
            std::vector<uint8_t> const& expected_level,
            std::vector<bool> const& expected_is_internal_node,
            std::vector<uint32_t> const& expected_length,
            std::vector<uint32_t> const& expected_offset)
  {
    thrust::device_vector<cuspatial::vec_2d<T>> d_points{points};

    auto [point_indices, tree] = cuspatial::quadtree_on_points(
      d_points.begin(), d_points.end(), vertex_1, vertex_2, scale, max_depth, max_size);

    EXPECT_EQ(point_indices.size(), points.size());

    auto& key_d              = tree.key;
    auto& level_d            = tree.level;
    auto& is_internal_node_d = tree.is_internal_node;
    auto& length_d           = tree.length;
    auto& offset_d           = tree.offset;

    EXPECT_EQ(key_d.size(), expected_key.size());
    EXPECT_EQ(level_d.size(), expected_level.size());
    EXPECT_EQ(is_internal_node_d.size(), expected_is_internal_node.size());
    EXPECT_EQ(length_d.size(), expected_length.size());
    EXPECT_EQ(offset_d.size(), expected_offset.size());

    using namespace cuspatial::test;

    expect_vector_equivalent(expected_key, key_d);
    expect_vector_equivalent(expected_level, level_d);
    expect_vector_equivalent(expected_is_internal_node, is_internal_node_d);
    expect_vector_equivalent(expected_length, length_d);
    expect_vector_equivalent(expected_offset, offset_d);
  }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(QuadtreeOnPointIndexingTest, TestTypes);

TYPED_TEST(QuadtreeOnPointIndexingTest, test_empty)
{
  using T = TypeParam;
  const cuspatial::vec_2d<T> vertex_1{0.0, 0.0};
  const cuspatial::vec_2d<T> vertex_2{1.0, 1.0};
  const int8_t max_depth = 1;
  const int32_t max_size = 1;
  const T scale          = 1.0;

  this->test({}, vertex_1, vertex_2, scale, max_depth, max_size, {}, {}, {}, {}, {});
}

TYPED_TEST(QuadtreeOnPointIndexingTest, test_single)
{
  using T = TypeParam;
  const cuspatial::vec_2d<T> vertex_1{0.0, 0.0};
  const cuspatial::vec_2d<T> vertex_2{1.0, 1.0};
  const int8_t max_depth = 1;
  const int32_t max_size = 1;
  const T scale          = 1.0;

  this->test(
    {{0.45, 0.45}}, vertex_1, vertex_2, scale, max_depth, max_size, {0}, {0}, {false}, {1}, {0});
}

TYPED_TEST(QuadtreeOnPointIndexingTest, test_two)
{
  using T = TypeParam;
  const cuspatial::vec_2d<T> vertex_1{0.0, 0.0};
  const cuspatial::vec_2d<T> vertex_2{2.0, 2.0};
  const int8_t max_depth = 1;
  const int32_t max_size = 1;
  const T scale          = 1.0;

  this->test({{0.45, 0.45}, {1.45, 1.45}},
             vertex_1,
             vertex_2,
             scale,
             max_depth,
             max_size,
             {0, 3},
             {0, 0},
             {false, false},
             {1, 1},
             {0, 1});
}

TYPED_TEST(QuadtreeOnPointIndexingTest, test_all_lowest_level_quads)
{
  using T = TypeParam;
  const cuspatial::vec_2d<T> vertex_1{-1000.0, -1000.0};
  const cuspatial::vec_2d<T> vertex_2{1000.0, 1000.0};
  const int8_t max_depth = 2;
  const int32_t max_size = 1;

  this->test({{-100.0, -100.0}, {100.0, 100.0}},
             vertex_1,
             vertex_2,
             -1,
             max_depth,
             max_size,
             {3, 12, 15},
             {0, 1, 1},
             {true, false, false},
             {2, 1, 1},
             {1, 0, 1});
}

TYPED_TEST(QuadtreeOnPointIndexingTest, test_small)
{
  using T = TypeParam;
  const cuspatial::vec_2d<T> vertex_1{0.0, 0.0};
  const cuspatial::vec_2d<T> vertex_2{8.0, 8.0};
  const int8_t max_depth = 3;
  const int32_t max_size = 12;
  const T scale          = 1.0;

  this->test(
    {
      {1.9804558865545805, 1.3472225743317712},  {0.1895259128530169, 0.5431061133894604},
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
      {3.154282922203257, 5.788316610960881},
    },
    vertex_1,
    vertex_2,
    scale,
    max_depth,
    max_size,
    {0, 1, 2, 0, 1, 3, 4, 7, 5, 6, 13, 14, 28, 31},
    {0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2},
    {true, true, false, false, true, true, false, true, false, false, false, false, false, false},
    {3, 2, 11, 7, 2, 2, 9, 2, 9, 7, 5, 8, 8, 7},
    {3, 6, 60, 0, 8, 10, 36, 12, 7, 16, 23, 28, 45, 53});
}
