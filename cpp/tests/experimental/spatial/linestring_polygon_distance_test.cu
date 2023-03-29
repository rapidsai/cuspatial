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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/linestring_polygon_distance.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct PairwiseLinestringPolygonDistanceTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::mr::device_memory_resource* mr() { return rmm::mr::get_current_device_resource(); }

  void run_single(std::initializer_list<std::size_t> multilinestring_geometry_offsets,
                  std::initializer_list<std::size_t> multilinestring_part_offsets,
                  std::initializer_list<vec_2d<T>> multilinestring_coordinates,
                  std::initializer_list<std::size_t> multipolygon_geometry_offsets,
                  std::initializer_list<std::size_t> multipolygon_part_offsets,
                  std::initializer_list<std::size_t> multipolygon_ring_offsets,
                  std::initializer_list<vec_2d<T>> multipolygon_coordinates,
                  std::initializer_list<T> expected)
  {
    auto multilinestrings = make_multilinestring_array(
      multilinestring_geometry_offsets, multilinestring_part_offsets, multilinestring_coordinates);

    auto multipolygons = make_multipolygon_array(multipolygon_geometry_offsets,
                                                 multipolygon_part_offsets,
                                                 multipolygon_ring_offsets,
                                                 multipolygon_coordinates);

    auto got = rmm::device_uvector<T>(multilinestrings.size(), stream());

    auto ret = pairwise_linestring_polygon_distance(
      multilinestrings.range(), multipolygons.range(), got.begin(), stream());

    auto d_expected = make_device_vector(expected);

    // CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(got, d_expected);
    // EXPECT_EQ(ret, got.end());
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwiseLinestringPolygonDistanceTest, TestTypes);

// Inputs are empty columns
TYPED_TEST(PairwiseLinestringPolygonDistanceTest, ZeroPairs)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0},
                     {0},
                     std::initializer_list<P>{},
                     {0},
                     {0},
                     {0},
                     std::initializer_list<P>{},
                     {});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, TwoPairs)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 1, 2},
                     {0, 4, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {0, 1, 2},
                     {0, 1, 2},
                     {0, 4, 9},
                     {P{-1, -1},
                      P{-2, -2},
                      P{-2, -1},
                      P{-1, -1},
                      P{-10, -10},
                      P{-10, -11},
                      P{-11, -11},
                      P{-11, -10},
                      P{-10, -10}},
                     {0.0, 0.0});
}
