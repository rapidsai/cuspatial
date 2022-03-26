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

#include <cuspatial/error.hpp>
#include <cuspatial/hausdorff.hpp>
#include <cuspatial/polygon_distance.hpp>

#include <cudf/column/column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <vector>

using namespace cudf;
using namespace test;

using TestTypes            = Types<float, double>;
using UnsupportedTestTypes = RemoveIf<ContainedIn<TestTypes>, NumericTypes>;

template <typename T>
struct PolygonDistanceTest : public BaseFixture {
};

TYPED_TEST_CASE(PolygonDistanceTest, TestTypes);

TYPED_TEST(PolygonDistanceTest, ZeroPoints)
{
  auto const xs = fixed_width_column_wrapper<TypeParam>{};
  auto const ys = fixed_width_column_wrapper<TypeParam>{};
  std::vector<size_type> h_space_offsets{};
  thrust::device_vector<size_type> d_space_offsests(h_space_offsets.begin(), h_space_offsets.end());

  auto const expected = fixed_width_column_wrapper<TypeParam>{};
  auto got            = cuspatial::polygon_distance(xs, ys, d_space_offsests);

  expect_columns_equal(expected, *got);

  // Ideally:
  // space_offsets = fixed_width_column_wrapper<size_type>{};
  // got = polygon_distance(xs, ys, column_view(space_offsets));
}

TYPED_TEST(PolygonDistanceTest, TwoPolygonsNoIntersect)
{
  // Polygon 1: (0, 0), (0, 2), (2, 0), (0, 0)
  // Polygon 2: (3, 1), (3, -1), (7, 5), (7, 1), (5, 2), (3, 1)
  // Distance: 1, from point (2, 0) to line segment (3, 1)->(3,-1)
  auto const xs = fixed_width_column_wrapper<TypeParam>{0, 0, 2, 0, 3, 3, 7, 7, 5, 3};
  auto const ys = fixed_width_column_wrapper<TypeParam>{0, 2, 0, 0, 1, -1, 5, 1, 2, 1};
  std::vector<size_type> h_space_offsets{0, 5};
  thrust::device_vector<size_type> d_space_offsests(h_space_offsets.begin(), h_space_offsets.end());

  auto const expected = fixed_width_column_wrapper<TypeParam>{0, 1, 1, 0};
  auto got            = cuspatial::polygon_distance(xs, ys, d_space_offsests);

  expect_columns_equal(expected, *got);
}

TYPED_TEST(PolygonDistanceTest, TwoPolygonsIntersect)
{
  // Polygon 1: (0, 0), (0, 2), (2, 0), (0, 0)
  // Polygon 2: (1, 1), (2, 1), (2, 2), (1, 2), (1, 1)
  // Distance: 0, two polygons intersects
  auto const xs = fixed_width_column_wrapper<TypeParam>{0, 0, 2, 0, 1, 2, 2, 1, 1};
  auto const ys = fixed_width_column_wrapper<TypeParam>{0, 2, 0, 0, 1, 1, 2, 2, 1};
  std::vector<size_type> h_space_offsets{0, 5};
  thrust::device_vector<size_type> d_space_offsests(h_space_offsets.begin(), h_space_offsets.end());

  auto const expected = fixed_width_column_wrapper<TypeParam>{0, 0, 0, 0};
  auto got            = cuspatial::polygon_distance(xs, ys, d_space_offsests);

  expect_columns_equal(expected, *got);
}

TYPED_TEST(PolygonDistanceTest, TwoPolygonsContainTest)
{
  // Polygon 1: (0, 0), (3, 0), (3, 3), (0, 3), (0, 0)
  // Polygon 2: (1, 1), (1, 2), (2, 2), (2, 1), (1, 1)
  // Distance: 0, polygon 1 contains polygon 2
  auto const xs = fixed_width_column_wrapper<TypeParam>{0, 0, 2, 0, 1, 2, 2, 1, 1};
  auto const ys = fixed_width_column_wrapper<TypeParam>{0, 2, 0, 0, 1, 1, 2, 2, 1};
  std::vector<size_type> h_space_offsets{0, 5};
  thrust::device_vector<size_type> d_space_offsests(h_space_offsets.begin(), h_space_offsets.end());

  auto const expected = fixed_width_column_wrapper<TypeParam>{0, 0, 0, 0};
  auto got            = cuspatial::polygon_distance(xs, ys, d_space_offsests);

  expect_columns_equal(expected, *got);
}
