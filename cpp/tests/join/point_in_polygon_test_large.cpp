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

#include <cuspatial/error.hpp>
#include <cuspatial/point_in_polygon.hpp>
#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/polygon_bounding_box.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/sort.h>

/*
 * The test uses the same quadtree structure as in pip_refine_test_small. However, the number of
 * randomly generated points under all quadrants (min_size) are increased to be more than the
 * number of threads per-block.
 */

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
struct PIPRefineTestLarge : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(PIPRefineTestLarge, cudf::test::FloatingPointTypes);

template <typename T>
inline auto generate_points(std::vector<std::vector<T>> const& quads, uint32_t points_per_quad)
{
  std::vector<T> point_x(quads.size() * points_per_quad);
  std::vector<T> point_y(quads.size() * points_per_quad);
  for (uint32_t i = 0, pos = 0; i < quads.size(); i++, pos += points_per_quad) {
    cudf::test::UniformRandomGenerator<T> dist_x{quads[i][0], quads[i][1]};
    cudf::test::UniformRandomGenerator<T> dist_y{quads[i][2], quads[i][3]};
    std::generate(point_x.begin() + pos, point_x.begin() + pos + points_per_quad, [&]() mutable {
      return dist_x.generate();
    });
    std::generate(point_y.begin() + pos, point_y.begin() + pos + points_per_quad, [&]() mutable {
      return dist_y.generate();
    });
  }
  return std::make_pair(std::move(point_x), std::move(point_y));
}

TYPED_TEST(PIPRefineTestLarge, TestLarge)
{
  using T = TypeParam;
  using namespace cudf::test;

  double const x_min{0.0};
  double const x_max{8.0};
  double const y_min{0.0};
  double const y_max{8.0};
  double const scale{1.0};
  uint32_t const max_depth{3};
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

  auto host_points = generate_points<T>(quads, min_size);
  auto& h_x        = std::get<0>(host_points);
  auto& h_y        = std::get<1>(host_points);
  auto x           = fixed_width_column_wrapper<T>(h_x.begin(), h_x.end());
  auto y           = fixed_width_column_wrapper<T>(h_y.begin(), h_y.end());

  auto quadtree_pair = cuspatial::quadtree_on_points(
    x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size, this->mr());

  auto& quadtree      = std::get<1>(quadtree_pair);
  auto& point_indices = std::get<0>(quadtree_pair);
  auto points         = cudf::gather(
    cudf::table_view{{x, y}}, *point_indices, cudf::out_of_bounds_policy::DONT_CHECK, this->mr());

  auto poly_offsets = fixed_width_column_wrapper<int32_t>({0, 1, 2, 3, 4});
  auto ring_offsets = fixed_width_column_wrapper<int32_t>({0, 4, 10, 14, 19});
  auto poly_x       = fixed_width_column_wrapper<T>({// ring 1
                                               2.488450,
                                               1.333584,
                                               3.460720,
                                               2.488450,
                                               // ring 2
                                               5.039823,
                                               5.561707,
                                               7.103516,
                                               7.190674,
                                               5.998939,
                                               5.039823,
                                               // ring 3
                                               5.998939,
                                               5.573720,
                                               6.703534,
                                               5.998939,
                                               // ring 4
                                               2.088115,
                                               1.034892,
                                               2.415080,
                                               3.208660,
                                               2.088115});
  auto poly_y       = fixed_width_column_wrapper<T>({// ring 1
                                               5.856625,
                                               5.008840,
                                               4.586599,
                                               5.856625,
                                               // ring 2
                                               4.229242,
                                               1.825073,
                                               1.503906,
                                               4.025879,
                                               5.653384,
                                               4.229242,
                                               // ring 3
                                               1.235638,
                                               0.197808,
                                               0.086693,
                                               1.235638,
                                               // ring 4
                                               4.541529,
                                               3.530299,
                                               2.896937,
                                               3.745936,
                                               4.541529});

  auto polygon_bboxes =
    cuspatial::polygon_bounding_boxes(poly_offsets, ring_offsets, poly_x, poly_y, 0.0, this->mr());

  auto polygon_quadrant_pairs = cuspatial::join_quadtree_and_bounding_boxes(
    *quadtree, *polygon_bboxes, x_min, x_max, y_min, y_max, scale, max_depth, this->mr());

  auto point_in_polygon_pairs = cuspatial::quadtree_point_in_polygon(*polygon_quadrant_pairs,
                                                                     *quadtree,
                                                                     *point_indices,
                                                                     x,
                                                                     y,
                                                                     poly_offsets,
                                                                     ring_offsets,
                                                                     poly_x,
                                                                     poly_y,
                                                                     this->mr());

  auto poly_idx  = point_in_polygon_pairs->get_column(0).view();
  auto point_idx = point_in_polygon_pairs->get_column(1).view();

  auto actual_poly_indices  = cudf::test::to_host<uint32_t>(poly_idx).first;
  auto actual_point_indices = cudf::test::to_host<uint32_t>(point_idx).first;

  thrust::stable_sort_by_key(
    actual_point_indices.begin(), actual_point_indices.end(), actual_poly_indices.begin());

  {
    // verify

    auto hits = cuspatial::point_in_polygon(
      points->get_column(0), points->get_column(1), poly_offsets, ring_offsets, poly_x, poly_y);

    auto hits_host = cudf::test::to_host<int32_t>(hits->view()).first;

    std::vector<uint32_t> expected_poly_indices;
    std::vector<uint32_t> expected_point_indices;

    for (int point_index = 0; point_index < hits->size(); point_index++) {
      std::uint32_t bits = hits_host[point_index];
      while (bits != 0) {
        std::uint32_t t          = bits & -bits;          // get only LSB
        std::uint32_t poly_index = __builtin_ctzl(bits);  // get index of LSB
        expected_poly_indices.push_back(poly_index);
        expected_point_indices.push_back(point_index);
        bits ^= t;  // reset LSB to zero to advance to next set bit
      }
    }

    thrust::stable_sort_by_key(
      expected_point_indices.begin(), expected_point_indices.end(), expected_poly_indices.begin());

    auto poly_a = fixed_width_column_wrapper<uint32_t>(expected_poly_indices.begin(),
                                                       expected_poly_indices.end());
    auto poly_b =
      fixed_width_column_wrapper<uint32_t>(actual_poly_indices.begin(), actual_poly_indices.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(poly_a, poly_b, verbosity);

    auto point_a = fixed_width_column_wrapper<uint32_t>(expected_point_indices.begin(),
                                                        expected_point_indices.end());
    auto point_b = fixed_width_column_wrapper<uint32_t>(actual_point_indices.begin(),
                                                        actual_point_indices.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(point_a, point_b, verbosity);
  }
}
