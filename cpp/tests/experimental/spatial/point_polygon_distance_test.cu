/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/experimental/point_polygon_distance.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct PairwisePointPolygonDistanceTest : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::mr::device_memory_resource* mr() { return rmm::mr::get_current_device_resource(); }

  void run_single(std::initializer_list<std::initializer_list<vec_2d<T>>> multipoints,
                  std::initializer_list<std::size_t> multipolygon_geometry_offsets,
                  std::initializer_list<std::size_t> multipolygon_part_offsets,
                  std::initializer_list<std::size_t> multipolygon_ring_offsets,
                  std::initializer_list<vec_2d<T>> multipolygon_coordinates,
                  std::initializer_list<T> expected)
  {
    auto d_multipoints   = make_multipoints_array(multipoints);
    auto d_multipolygons = make_multipolygon_array(multipolygon_geometry_offsets,
                                                   multipolygon_part_offsets,
                                                   multipolygon_ring_offsets,
                                                   multipolygon_coordinates);

    auto got = rmm::device_uvector<T>(d_multipoints.size(), stream());

    auto ret = pairwise_point_polygon_distance(
      d_multipoints.range(), d_multipolygons.range(), got.begin(), stream());

    auto d_expected = make_device_vector(expected);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(got, d_expected);
    EXPECT_EQ(ret, got.end());
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwisePointPolygonDistanceTest, TestTypes);

TYPED_TEST(PairwisePointPolygonDistanceTest, ZeroPairs)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     std::initializer_list<std::initializer_list<P>>{},
                     {0},
                     {0},
                     {0},
                     std::initializer_list<P>{},
                     std::initializer_list<T>{});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairOnePolygonOneRing)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{0, 0}}},
                     {0, 1},
                     {0, 1},
                     {0, 5},
                     {P{-1, -1}, P{1, -1}, P{1, 1}, P{-1, 1}, P{-1, -1}},
                     {0.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairOnePolygonOneRing2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{0, 2}}},
                     {0, 1},
                     {0, 1},
                     {0, 5},
                     {P{-1, -1}, P{1, -1}, P{1, 1}, P{-1, 1}, P{-1, -1}},
                     {1.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairOnePolygonTwoRings)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{0, 0}}},
                     {0, 1},
                     {0, 2},
                     {0, 5, 10},
                     {
                       P{-2, -2},
                       P{2, -2},
                       P{2, 2},
                       P{-2, 2},
                       P{-2, -2},
                       P{-1, -1},
                       P{1, -1},
                       P{1, 1},
                       P{-1, 1},
                       P{-1, -1},
                     },
                     {1.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairOnePolygonTwoRings2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{1.5, 0}}},
                     {0, 1},
                     {0, 2},
                     {0, 5, 10},
                     {
                       P{-2, -2},
                       P{2, -2},
                       P{2, 2},
                       P{-2, 2},
                       P{-2, -2},
                       P{-1, -1},
                       P{1, -1},
                       P{1, 1},
                       P{-1, 1},
                       P{-1, -1},
                     },
                     {0.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairOnePolygonTwoRings3)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{3, 0}}},
                     {0, 1},
                     {0, 2},
                     {0, 5, 10},
                     {
                       P{-2, -2},
                       P{2, -2},
                       P{2, 2},
                       P{-2, 2},
                       P{-2, -2},
                       P{-1, -1},
                       P{1, -1},
                       P{1, 1},
                       P{-1, 1},
                       P{-1, -1},
                     },
                     {1.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairTwoPolygonOneRing)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{1, 1}}},
                     {0, 2},
                     {0, 1, 2},
                     {0, 5, 10},
                     {
                       P{-2, -2},
                       P{0, -2},
                       P{0, 0},
                       P{-2, 0},
                       P{-2, -2},
                       P{0, 0},
                       P{2, 0},
                       P{2, 2},
                       P{0, 2},
                       P{0, 0},
                     },
                     {0.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairTwoPolygonOneRing2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{-1, -1}}},
                     {0, 2},
                     {0, 1, 2},
                     {0, 5, 10},
                     {
                       P{-2, -2},
                       P{0, -2},
                       P{0, 0},
                       P{-2, 0},
                       P{-2, -2},
                       P{0, 0},
                       P{2, 0},
                       P{2, 2},
                       P{0, 2},
                       P{0, 0},
                     },
                     {0.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairTwoPolygonOneRing3)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{-1, 0.5}}},
                     {0, 2},
                     {0, 1, 2},
                     {0, 5, 10},
                     {
                       P{-2, -2},
                       P{0, -2},
                       P{0, 0},
                       P{-2, 0},
                       P{-2, -2},
                       P{0, 0},
                       P{2, 0},
                       P{2, 2},
                       P{0, 2},
                       P{0, 0},
                     },
                     {0.5});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairTwoPolygonOneRing4)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{-0.3, 1}}},
                     {0, 2},
                     {0, 1, 2},
                     {0, 5, 10},
                     {
                       P{-2, -2},
                       P{0, -2},
                       P{0, 0},
                       P{-2, 0},
                       P{-2, -2},
                       P{0, 0},
                       P{2, 0},
                       P{2, 2},
                       P{0, 2},
                       P{0, 0},
                     },
                     {0.3});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, TwoPairOnePolygonOneRing)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{-0.6, -0.6}}, {P{0, 0}}},
                     {0, 1, 2},
                     {0, 1, 2},
                     {0, 4, 8},
                     {
                       P{-1, -1},
                       P{0, 0},
                       P{0, 1},
                       P{-1, -1},
                       P{1, 1},
                       P{1, 0},
                       P{2, 2},
                       P{1, 1},
                     },
                     {0.0, 1.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, TwoPairTwoPolygonTwoRing)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{2.5, 3}}, {P{-1.75, -1.5}}},
                     {0, 1, 2},
                     {0, 2, 4},
                     {0, 5, 10, 14, 18},
                     {
                       P{0, 0},
                       P{3, 0},
                       P{3, 3},
                       P{0, 3},
                       P{0, 0},
                       P{1, 1},
                       P{2, 1},
                       P{2, 2},
                       P{1, 2},
                       P{1, 1},

                       P{0, 0},
                       P{-3, 0},
                       P{-3, -3},
                       P{0, 0},
                       P{-1, -1},
                       P{-2, -1},
                       P{-2, -2},
                       P{-1, -1},
                     },
                     {0.0, 0.17677669529663687});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, ThreePolygons)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{1, 1}}, {P{2, 2}}, {P{1.5, 0}}},
                     {0, 1, 2, 3},
                     {0, 2, 3, 6},
                     {0, 4, 8, 12, 17, 22, 27},
                     {// POLYGON ((0 1, -1 -1, 1 -1, 0 1), (0 0.5, 0.5 -0.5, -0.5 -0.5, 0 0.5))
                      P{0, 1},
                      P{-1, -1},
                      P{1, -1},
                      P{0, 1},
                      P{0, 0.5},
                      P{0.5, -0.5},
                      P{-0.5, -0.5},
                      P{0, 0.5},
                      // POLYGON ((1 1, 1 2, 2 1, 1 1))
                      P{1, 1},
                      P{1, 2},
                      P{2, 1},
                      P{1, 1},
                      //  POLYGON (
                      //      (-3 -3, 3 -3, 3 3, -3 3, -3 -3),
                      //      (-2 -2, -1 -2, -1 2, -2 2, -2 -2),
                      //      (2 2, 2 -2, 1 -2, 1 2, 2 2)
                      //  )
                      P{-3, -3},
                      P{3, -3},
                      P{3, 3},
                      P{-3, 3},
                      P{-3, -3},
                      P{-2, -2},
                      P{-1, -2},
                      P{-1, 2},
                      P{-2, 2},
                      P{-2, -2},
                      P{2, 2},
                      P{2, -2},
                      P{1, -2},
                      P{1, 2},
                      P{2, 2}},
                     {0.894427190999916, 0.7071067811865476, 0.5});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairMultiPointOnePolygon)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{0, 3}, P{2, 0}}},
                     {0, 1},
                     {0, 1},
                     {0, 5},
                     {P{0, 1}, P{-1, -1}, P{1, -1}, P{0, 1}},
                     {1.3416407864998738});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, OnePairMultiPointOnePolygon2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {{P{0, 3}, P{0, 0}}},
                     {0, 1},
                     {0, 1},
                     {0, 5},
                     {P{0, 1}, P{-1, -1}, P{1, -1}, P{0, 1}},
                     {0.0});
}

TYPED_TEST(PairwisePointPolygonDistanceTest, TwoPairMultiPointOnePolygon2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(
    this->run_single,
    {{P{0, 2}, P{0, 0}}, {P{1, 1}, P{-1, -1}}},
    {0, 1, 2},
    {0, 1, 2},
    {0, 5, 9},
    {P{-1, -1}, P{1, -1}, P{1, 1}, P{-1, 1}, P{-1, -1}, P{-1, 1}, P{1, 1}, P{0, -1}, P{-1, 1}},
    {0.0, 0.0});
}
