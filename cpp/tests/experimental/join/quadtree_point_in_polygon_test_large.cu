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
#include <cuspatial_test/random.cuh>
#include <cuspatial_test/test_util.cuh>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/error.hpp>
#include <cuspatial/geometry/box.hpp>
#include <cuspatial/point_in_polygon.cuh>
#include <cuspatial/polygon_bounding_boxes.cuh>
#include <cuspatial/spatial_join.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <thrust/gather.h>
#include <thrust/sort.h>

/*
 * The test uses the same quadtree structure as in pip_refine_test_small. However, the number of
 * randomly generated points under all quadrants (min_size) are increased to be more than the
 * number of threads per-block.
 */

template <typename T>
struct PIPRefineTestLarge : public cuspatial::test::BaseFixture {};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PIPRefineTestLarge, TestTypes);

template <typename T>
inline auto generate_points(
  std::vector<std::vector<T>> const& quads,
  uint32_t points_per_quad,
  std::size_t seed,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto engine     = cuspatial::test::deterministic_engine(0);
  auto uniform    = cuspatial::test::make_normal_dist<T>(0.0, 1.0);
  auto pgen       = cuspatial::test::point_generator(T{0.0}, T{1.0}, engine, uniform);
  auto num_points = quads.size() * points_per_quad;
  rmm::device_uvector<cuspatial::vec_2d<T>> points(num_points, stream, mr);

  auto counting_iter = thrust::make_counting_iterator(seed);
  thrust::transform(
    rmm::exec_policy(stream), counting_iter, counting_iter + num_points, points.begin(), pgen);

  return points;
}

TYPED_TEST(PIPRefineTestLarge, TestLarge)
{
  using T = TypeParam;
  using cuspatial::vec_2d;
  using cuspatial::test::make_device_vector;

  vec_2d<T> v_min{0.0, 0.0};
  vec_2d<T> v_max{8.0, 8.0};
  T const scale{1.0};
  uint8_t const max_depth{3};
  uint32_t const min_size{400};

  std::vector<std::vector<T>> quads{{0, 2, 0, 2},
                                    {3, 4, 0, 1},
                                    {2, 3, 1, 2},
                                    {4, 6, 0, 2},
                                    {3, 4, 2, 3},
                                    {2, 3, 3, 4},
                                    {6, 7, 2, 3},
                                    {7, 8, 3, 4},
                                    {0, 4, 4, 8}};

  auto points_in = generate_points<T>(quads, min_size, 0, this->stream());

  auto [point_indices, quadtree] = quadtree_on_points(
    points_in.begin(), points_in.end(), v_min, v_max, scale, max_depth, min_size, this->stream());

  auto points = rmm::device_uvector<vec_2d<T>>(quads.size() * min_size, this->stream());
  thrust::gather(rmm::exec_policy(this->stream()),
                 point_indices.begin(),
                 point_indices.end(),
                 points_in.begin(),
                 points.begin());

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

  thrust::stable_sort_by_key(rmm::exec_policy(this->stream()),
                             actual_point_indices.begin(),
                             actual_point_indices.end(),
                             actual_poly_indices.begin());

  {  // verify
    rmm::device_uvector<int32_t> hits(points.size(), this->stream());
    auto hits_end = cuspatial::point_in_polygon(points.begin(),
                                                points.end(),
                                                multipolygons.part_offset_begin(),
                                                multipolygons.part_offset_end(),
                                                multipolygons.ring_offset_begin(),
                                                multipolygons.ring_offset_end(),
                                                multipolygons.point_begin(),
                                                multipolygons.point_end(),
                                                hits.begin(),
                                                this->stream());

    auto hits_host = cuspatial::test::to_host<int32_t>(hits);

    std::vector<uint32_t> expected_poly_indices;
    std::vector<uint32_t> expected_point_indices;

    for (std::size_t point_index = 0; point_index < hits_host.size(); point_index++) {
      // iterate over set bits
      std::uint32_t bits = hits_host[point_index];
      while (bits != 0) {
        std::uint32_t t          = bits & -bits;         // get only LSB
        std::uint32_t poly_index = __builtin_ctz(bits);  // get index of LSB
        expected_poly_indices.push_back(poly_index);
        expected_point_indices.push_back(point_index);
        bits ^= t;  // reset LSB to zero to advance to next set bit
      }
    }

    // TODO: shouldn't have to copy to device here, but I get Thrust compilation errors if I use
    // host vectors and a host stable_sort_by_key.
    auto d_expected_poly_indices  = rmm::device_vector<std::uint32_t>(expected_poly_indices);
    auto d_expected_point_indices = rmm::device_vector<std::uint32_t>(expected_point_indices);

    thrust::stable_sort_by_key(rmm::exec_policy(this->stream()),
                               d_expected_point_indices.begin(),
                               d_expected_point_indices.end(),
                               d_expected_poly_indices.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected_poly_indices, actual_poly_indices);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected_point_indices, actual_point_indices);
  }
}
