/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cuspatial/error.hpp>
#include <cuspatial/point_quadtree.cuh>
#include <cuspatial/spatial_join.cuh>

// Note: the detailed correctness test of the join_quadtree_and_bounding_boxes() function is covered
// by the quadtree_point_in_polygon_test_small.cu test file.

template <typename T>
struct JoinQuadtreeAndBoundingBoxesErrorTest : public cuspatial::test::BaseFixture {};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(JoinQuadtreeAndBoundingBoxesErrorTest, TestTypes);

TYPED_TEST(JoinQuadtreeAndBoundingBoxesErrorTest, test_empty)
{
  using T = TypeParam;

  using namespace cuspatial;
  using cuspatial::vec_2d;
  using cuspatial::test::make_device_vector;

  vec_2d<T> v_min{0.0, 0.0};
  T const scale{1.0};
  uint8_t const max_depth{1};

  auto empty_quadtree = point_quadtree_ref{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  auto empty_bboxes = rmm::device_uvector<box<T>>{0, this->stream()};

  auto [polygon_idx, quad_idx] = cuspatial::join_quadtree_and_bounding_boxes(empty_quadtree,
                                                                             empty_bboxes.begin(),
                                                                             empty_bboxes.end(),
                                                                             v_min,
                                                                             scale,
                                                                             max_depth,
                                                                             this->stream());

  auto expected_polygon_idx = rmm::device_uvector<std::uint32_t>{0, this->stream()};
  auto expected_quad_idx    = rmm::device_uvector<std::uint32_t>{0, this->stream()};

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_polygon_idx, polygon_idx);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_quad_idx, quad_idx);
}
