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
#include <cuspatial/experimental/polygon_bounding_boxes.cuh>
#include <cuspatial/vec_2d.hpp>

#include <gtest/gtest.h>

template <typename T>
struct PolygonBoundingBoxTest : public ::testing::Test {
};

using cuspatial::vec_2d;

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PolygonBoundingBoxTest, TestTypes);

TYPED_TEST(PolygonBoundingBoxTest, test_empty)
{
  using T = TypeParam;

  auto poly_offsets = make_device_vector<int32_t>({});
  auto ring_offsets = make_device_vector<int32_t>({});
  auto vertices     = make_device_vector<vec_2d<T>>({});

  auto bbox_min = rmm::device_vector<cuspatial::vec_2d<T>>(poly_offsets.size());
  auto bbox_max = rmm::device_vector<cuspatial::vec_2d<T>>(poly_offsets.size());

  auto bboxes_begin = thrust::make_zip_iterator(bbox_min.begin(), bbox_max.begin());

  auto bboxes_end = cuspatial::polygon_bounding_boxes(poly_offsets.begin(),
                                                      poly_offsets.end(),
                                                      ring_offsets.begin(),
                                                      ring_offsets.end(),
                                                      vertices.begin(),
                                                      vertices.end(),
                                                      bboxes_begin);

  EXPECT_EQ(std::distance(bboxes_begin, bboxes_end), 0);
}

TYPED_TEST(PolygonBoundingBoxTest, test_one)
{
  using T = TypeParam;

  auto poly_offsets = make_device_vector<int32_t>({0});
  auto ring_offsets = make_device_vector<int32_t>({0});
  auto vertices     = make_device_vector<vec_2d<T>>(
    {{2.488450, 5.856625}, {1.333584, 5.008840}, {3.460720, 4.586599}, {2.488450, 5.856625}});

  auto bbox_min = rmm::device_vector<cuspatial::vec_2d<T>>(poly_offsets.size());
  auto bbox_max = rmm::device_vector<cuspatial::vec_2d<T>>(poly_offsets.size());

  auto bboxes_begin = thrust::make_zip_iterator(bbox_min.begin(), bbox_max.begin());

  auto bboxes_end = cuspatial::polygon_bounding_boxes(poly_offsets.begin(),
                                                      poly_offsets.end(),
                                                      ring_offsets.begin(),
                                                      ring_offsets.end(),
                                                      vertices.begin(),
                                                      vertices.end(),
                                                      bboxes_begin);

  EXPECT_EQ(std::distance(bboxes_begin, bboxes_end), 1);

  auto bbox_min_expected = make_device_vector<vec_2d<T>>({{1.333584, 4.586599}});
  auto bbox_max_expected = make_device_vector<vec_2d<T>>({{3.460720, 5.856625}});
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(bbox_min, bbox_min_expected);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(bbox_max, bbox_max_expected);
}

TYPED_TEST(PolygonBoundingBoxTest, test_small)
{
  using T = TypeParam;

  auto poly_offsets = make_device_vector<int32_t>({0, 1, 2, 3});
  auto ring_offsets = make_device_vector<int32_t>({0, 4, 10, 14});
  auto vertices     = make_device_vector<vec_2d<T>>({// ring 1
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

  auto bbox_min = rmm::device_vector<cuspatial::vec_2d<T>>(poly_offsets.size());
  auto bbox_max = rmm::device_vector<cuspatial::vec_2d<T>>(poly_offsets.size());

  auto bboxes_begin = thrust::make_zip_iterator(bbox_min.begin(), bbox_max.begin());

  auto bboxes_end = cuspatial::polygon_bounding_boxes(poly_offsets.begin(),
                                                      poly_offsets.end(),
                                                      ring_offsets.begin(),
                                                      ring_offsets.end(),
                                                      vertices.begin(),
                                                      vertices.end(),
                                                      bboxes_begin);

  EXPECT_EQ(std::distance(bboxes_begin, bboxes_end), 4);

  auto bbox_min_expected =
    make_device_vector<vec_2d<T>>({{1.3335840000000001, 4.5865989999999996},
                                   {5.0398230000000002, 1.503906},
                                   {5.5737199999999998, 0.086693000000000006},
                                   {1.0348919999999999, 2.8969369999999999}});
  auto bbox_max_expected =
    make_device_vector<vec_2d<T>>({{3.4607199999999998, 5.8566250000000002},
                                   {7.1906739999999996, 5.653384},
                                   {6.7035340000000003, 1.235638},
                                   {3.2086600000000001, 4.5415289999999997}});
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(bbox_min, bbox_min_expected);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(bbox_max, bbox_max_expected);
}
