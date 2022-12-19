/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/polygon_bounding_box.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/distance.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <ogrsf_frmts.h>

// #include <algorithm>

/*
 * The test uses the same quadtree structure as in pip_refine_test_small.
 * However, the numbers of randomly generated points under all quadrants (min_size) are increased
 * to be more than the number of threads per-block (currently fixed to 256, but can be set between
 * 32 2048 (CUDA Compute Capacity 7.0, multiples of warp size, which is 32) The test is designed to
 * fully test the two kernels in the refinment code, including both warp level reduce and scan, vote
 * and popc. Thrust primitives to divide quadrants into sub-blocks are also tested.
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

template <typename T>
inline auto make_polygons_geometry(thrust::host_vector<uint32_t> const& poly_offsets,
                                   thrust::host_vector<uint32_t> const& ring_offsets,
                                   thrust::host_vector<T> const& poly_x,
                                   thrust::host_vector<T> const& poly_y)
{
  std::vector<OGRGeometry*> polygons{};
  for (uint32_t poly_idx = 0, poly_end = poly_offsets.size() - 1; poly_idx < poly_end; ++poly_idx) {
    auto ring_idx = static_cast<size_t>(poly_offsets[poly_idx]);
    auto ring_end = static_cast<size_t>(poly_offsets[poly_idx + 1]);
    auto polygon  = static_cast<OGRPolygon*>(OGRGeometryFactory::createGeometry(wkbPolygon));
    for (; ring_idx < ring_end; ++ring_idx) {
      auto seg_idx = static_cast<size_t>(ring_offsets[ring_idx]);
      auto seg_end = static_cast<size_t>(
        ring_idx < ring_offsets.size() - 1 ? ring_offsets[ring_idx + 1] : poly_x.size());
      auto ring = static_cast<OGRLineString*>(OGRGeometryFactory::createGeometry(wkbLinearRing));
      for (; seg_idx < seg_end; ++seg_idx) {
        ring->addPoint(poly_x[seg_idx], poly_y[seg_idx]);
      }
      polygon->addRing(ring);
    }
    polygons.push_back(polygon);
  }
  return polygons;
}

template <typename T>
auto geometry_to_poly_and_point_indices(std::vector<OGRGeometry*> const& polygons,
                                        thrust::host_vector<T> const& x,
                                        thrust::host_vector<T> const& y)
{
  std::vector<uint32_t> poly_indices{};
  std::vector<uint32_t> point_lengths{};
  std::vector<uint32_t> point_indices{};

  for (uint32_t i = 0, n = x.size(); i < n; i++) {
    OGRPoint point(x[i], y[i]);
    std::vector<uint32_t> found_poly_idxs{};
    for (uint32_t j = 0; j < polygons.size(); j++) {
      if (polygons[j]->Contains(&point)) { found_poly_idxs.push_back(j); }
    }
    if (found_poly_idxs.size() > 0) {
      point_lengths.push_back(found_poly_idxs.size());
      point_indices.push_back(i);
      poly_indices.insert(poly_indices.end(), found_poly_idxs.begin(), found_poly_idxs.end());
    }
  }
  return std::make_tuple(
    std::move(poly_indices), std::move(point_indices), std::move(point_lengths));
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

  // verify

  auto h_poly = make_polygons_geometry(cudf::test::to_host<uint32_t>(poly_offsets).first,
                                       cudf::test::to_host<uint32_t>(ring_offsets).first,
                                       cudf::test::to_host<T>(poly_x).first,
                                       cudf::test::to_host<T>(poly_y).first);

  auto host_poly_and_point_indices =
    geometry_to_poly_and_point_indices(h_poly,
                                       cudf::test::to_host<T>(points->get_column(0)).first,
                                       cudf::test::to_host<T>(points->get_column(1)).first);

  auto& expected_poly_indices  = std::get<0>(host_poly_and_point_indices);
  auto& expected_point_indices = std::get<1>(host_poly_and_point_indices);
  auto& expected_point_lengths = std::get<2>(host_poly_and_point_indices);

  auto actual_poly_indices  = cudf::test::to_host<uint32_t>(poly_idx).first;
  auto actual_point_indices = cudf::test::to_host<uint32_t>(point_idx).first;
  auto actual_point_lengths = thrust::host_vector<uint32_t>(point_in_polygon_pairs->num_rows());

  thrust::stable_sort_by_key(
    actual_point_indices.begin(), actual_point_indices.end(), actual_poly_indices.begin());

  auto num_search_points = thrust::distance(actual_point_indices.begin(),
                                            thrust::reduce_by_key(actual_point_indices.begin(),
                                                                  actual_point_indices.end(),
                                                                  thrust::make_constant_iterator(1),
                                                                  actual_point_indices.begin(),
                                                                  actual_point_lengths.begin())
                                              .first);

  actual_point_indices.resize(num_search_points);
  actual_point_lengths.resize(num_search_points);
  actual_point_indices.shrink_to_fit();
  actual_point_lengths.shrink_to_fit();

  auto poly_a = fixed_width_column_wrapper<uint32_t>(expected_poly_indices.begin(),
                                                     expected_poly_indices.end());
  auto poly_b =
    fixed_width_column_wrapper<uint32_t>(actual_poly_indices.begin(), actual_poly_indices.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(poly_a, poly_b, verbosity);

  auto point_a = fixed_width_column_wrapper<uint32_t>(expected_point_indices.begin(),
                                                      expected_point_indices.end());
  auto point_b =
    fixed_width_column_wrapper<uint32_t>(actual_point_indices.begin(), actual_point_indices.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(point_a, point_b, verbosity);

  auto lengths_a = fixed_width_column_wrapper<uint32_t>(expected_point_lengths.begin(),
                                                        expected_point_lengths.end());
  auto lengths_b =
    fixed_width_column_wrapper<uint32_t>(actual_point_lengths.begin(), actual_point_lengths.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(lengths_a, lengths_b, verbosity);
}
