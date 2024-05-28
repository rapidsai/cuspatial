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
#include <cuspatial_test/geometry_generator.cuh>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/bounding_boxes.cuh>
#include <cuspatial/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/geometry/box.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/spatial_join.cuh>

#include <gtest/gtest.h>

/*
 * The test uses the same quadtree structure as in pip_refine_test_small. However, the number of
 * randomly generated points under all quadrants (min_size) are increased to be more than the
 * number of threads per-block.
 */

template <typename T>
struct PIPRefineTestLarge : public cuspatial::test::BaseFixture {};

template <typename T>
struct test_point_in_poly_functor {
  cuspatial::multipolygon_range<std::size_t*, std::size_t*, std::size_t*, cuspatial::vec_2d<T>*>
    polys;
  __device__ inline T operator()(cuspatial::vec_2d<T> point)
  {
    auto it = cuspatial::make_counting_transform_iterator(0, [&](auto i) { return polys[i][0]; });
    return thrust::count_if(thrust::seq, it, it + polys.num_multipolygons(), [&](auto poly) {
      return cuspatial::is_point_in_polygon(point, poly);
    });
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PIPRefineTestLarge, TestTypes);

TYPED_TEST(PIPRefineTestLarge, TestOOM)
{
  using T = TypeParam;
  using cuspatial::vec_2d;
  using cuspatial::test::make_device_vector;

  using point_t = typename cuspatial::test::multipoint_array<rmm::device_uvector<std::size_t>,
                                                             rmm::device_uvector<vec_2d<T>>>;

  using polys_t = typename cuspatial::test::multipolygon_array<rmm::device_uvector<std::size_t>,
                                                               rmm::device_uvector<std::size_t>,
                                                               rmm::device_uvector<std::size_t>,
                                                               rmm::device_uvector<vec_2d<T>>>;

  vec_2d<T> v_min{0.0, 0.0};
  vec_2d<T> v_max{1'000.0, 1'000.0};
  T const scale{1.0};
  std::uint8_t const max_depth{15};
  std::uint32_t const min_size{10'000};

  std::size_t const num_points{1'000'000};
  std::size_t const num_polys{24'000};

  rmm::device_uvector<vec_2d<T>> points = [&]() {
    point_t points = cuspatial::test::generate_multipoint_array(
      cuspatial::test::multipoint_generator_parameter<T>{1, num_points, v_min, v_max},
      this->stream());
    return points.release().second;
  }();

  polys_t multipoly_array = cuspatial::test::generate_multipolygon_array(
    cuspatial::test::multipolygon_generator_parameter<T>{
      num_polys,
      1,
      0,
      4,
      vec_2d<T>{(v_max - v_min).x / 2, (v_max - v_min).y / 2},
      (v_max - v_min).x / 8},
    this->stream());

  auto multipolygons = multipoly_array.range();

  auto expected_size = thrust::count_if(rmm::exec_policy(this->stream()),
                                        points.begin(),
                                        points.end(),
                                        test_point_in_poly_functor<T>{multipolygons});

  auto [point_indices, quadtree] = cuspatial::quadtree_on_points(
    points.begin(), points.end(), v_min, v_max, scale, max_depth, min_size, this->stream());

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

  EXPECT_GT(bboxes.size(), 0);

  auto [poly_indices, quad_indices] = cuspatial::join_quadtree_and_bounding_boxes(
    quadtree, bboxes.begin(), bboxes.end(), v_min, scale, max_depth, this->stream());

  auto [actual_poly_indices, actual_point_indices] =
    cuspatial::quadtree_point_in_polygon(poly_indices.begin(),
                                         poly_indices.end(),
                                         quad_indices.begin(),
                                         quadtree,
                                         point_indices.begin(),
                                         point_indices.end(),
                                         points.begin(),
                                         multipolygons,
                                         this->stream());

  EXPECT_GT(actual_point_indices.size(), 0);
  EXPECT_GT(actual_poly_indices.size(), 0);

  EXPECT_EQ(actual_point_indices.size(), expected_size);
  EXPECT_EQ(actual_poly_indices.size(), expected_size);
}
