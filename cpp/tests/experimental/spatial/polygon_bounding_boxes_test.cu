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
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/geometry/box.hpp>
#include <cuspatial/experimental/polygon_bounding_boxes.cuh>
#include <cuspatial/vec_2d.hpp>

#include <gtest/gtest.h>

template <typename T>
struct PolygonBoundingBoxTest : public ::testing::Test {
};

using cuspatial::vec_2d;
using cuspatial::test::make_device_vector;

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PolygonBoundingBoxTest, TestTypes);

TYPED_TEST(PolygonBoundingBoxTest, test_empty)
{
  using T = TypeParam;

  auto poly_offsets = make_device_vector<int32_t>({});
  auto ring_offsets = make_device_vector<int32_t>({});
  auto vertices     = make_device_vector<vec_2d<T>>({});

  auto bboxes = rmm::device_vector<cuspatial::box<T>>(poly_offsets.size());

  auto bboxes_end = cuspatial::polygon_bounding_boxes(poly_offsets.begin(),
                                                      poly_offsets.end(),
                                                      ring_offsets.begin(),
                                                      ring_offsets.end(),
                                                      vertices.begin(),
                                                      vertices.end(),
                                                      bboxes.begin());

  EXPECT_EQ(std::distance(bboxes.begin(), bboxes_end), 0);
}

TYPED_TEST(PolygonBoundingBoxTest, test_one)
{
  using T = TypeParam;

  // GeoArrow: Final offset points to the end of the data. The number of offsets is number of
  // geometries / parts plus one.
  auto poly_offsets = make_device_vector<int32_t>({0, 1});
  auto ring_offsets = make_device_vector<int32_t>({0, 4});
  auto vertices     = make_device_vector<vec_2d<T>>(
    {{2.488450, 5.856625}, {1.333584, 5.008840}, {3.460720, 4.586599}, {2.488450, 5.856625}});

  // GeoArrow: Number of linestrings is number of offsets minus one.
  auto bboxes = rmm::device_vector<cuspatial::box<T>>(poly_offsets.size() - 1);

  auto bboxes_end = cuspatial::polygon_bounding_boxes(poly_offsets.begin(),
                                                      poly_offsets.end(),
                                                      ring_offsets.begin(),
                                                      ring_offsets.end(),
                                                      vertices.begin(),
                                                      vertices.end(),
                                                      bboxes.begin());

  EXPECT_EQ(std::distance(bboxes.begin(), bboxes_end), 1);

  auto bboxes_expected =
    make_device_vector<cuspatial::box<T>>({{{1.333584, 4.586599}, {3.460720, 5.856625}}});

  cuspatial::test::expect_vec_2d_pair_equivalent(bboxes, bboxes_expected);
}

TYPED_TEST(PolygonBoundingBoxTest, test_small)
{
  using T = TypeParam;

  // GeoArrow: Final offset points to the end of the data. The number of offsets is number of
  // geometries / parts plus one.
  auto poly_offsets = make_device_vector<int32_t>({0, 1, 2, 3, 4});
  auto ring_offsets = make_device_vector<int32_t>({0, 4, 9, 13, 17});
  auto vertices     = make_device_vector<vec_2d<T>>({// ring 1 (closed)
                                                 {2.488450, 5.856625},
                                                 {1.333584, 5.008840},
                                                 {3.460720, 4.586599},
                                                 {2.488450, 5.856625},
                                                 // ring 2 (open)
                                                 {5.039823, 4.229242},
                                                 {5.561707, 1.825073},
                                                 {7.103516, 1.503906},
                                                 {7.190674, 4.025879},
                                                 {5.998939, 5.653384},
                                                 // ring 3 (closed)
                                                 {5.998939, 1.235638},
                                                 {5.573720, 0.197808},
                                                 {6.703534, 0.086693},
                                                 {5.998939, 1.235638},
                                                 // ring 4 (open)
                                                 {2.088115, 4.541529},
                                                 {1.034892, 3.530299},
                                                 {2.415080, 2.896937},
                                                 {3.208660, 3.745936}});

  // GeoArrow: Number of linestrings is number of offsets minus one.
  auto bboxes = rmm::device_vector<cuspatial::box<T>>(poly_offsets.size() - 1);

  auto bboxes_end = cuspatial::polygon_bounding_boxes(poly_offsets.begin(),
                                                      poly_offsets.end(),
                                                      ring_offsets.begin(),
                                                      ring_offsets.end(),
                                                      vertices.begin(),
                                                      vertices.end(),
                                                      bboxes.begin());

  EXPECT_EQ(std::distance(bboxes.begin(), bboxes_end), 4);

  auto bboxes_expected = make_device_vector<cuspatial::box<T>>(
    {{{1.3335840000000001, 4.5865989999999996}, {3.4607199999999998, 5.8566250000000002}},
     {{5.0398230000000002, 1.503906}, {7.1906739999999996, 5.653384}},
     {{5.5737199999999998, 0.086693000000000006}, {6.7035340000000003, 1.235638}},
     {{1.0348919999999999, 2.8969369999999999}, {3.2086600000000001, 4.5415289999999997}}});

  cuspatial::test::expect_vec_2d_pair_equivalent(bboxes, bboxes_expected);
}
