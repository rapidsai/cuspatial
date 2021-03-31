/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <indexing/construction/detail/utilities.cuh>
#include <utility/point_to_nearest_polyline.cuh>

#include <cuspatial/error.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/transform.h>

namespace cuspatial {
namespace detail {
namespace {

template <typename T, typename PointIter, typename QuadOffsetsIter>
struct compute_point_poly_indices_and_distances {
  PointIter points;
  QuadOffsetsIter quad_point_offsets;
  uint32_t const *local_point_offsets;
  size_t const num_local_point_offsets;
  cudf::column_device_view const poly_indices;
  cudf::column_device_view const poly_offsets;
  cudf::column_device_view const poly_points_x;
  cudf::column_device_view const poly_points_y;
  thrust::tuple<uint32_t, uint32_t, T> __device__ operator()(cudf::size_type const i)
  {
    // Calculate the position in "local_point_offsets" that `i` falls between.
    // This position is the index of the poly/quad pair for this `i`.
    //
    // Dereferencing `local_point_offset` yields the zero-based first point position of this
    // quadrant. Adding this zero-based position to the quadrant's first point position in the
    // quadtree yields the "global" position in the `point_indices` map.
    auto po_begin                 = local_point_offsets;
    auto po_end                   = local_point_offsets + num_local_point_offsets;
    auto const local_point_offset = thrust::upper_bound(thrust::seq, po_begin, po_end, i) - 1;
    uint32_t const pairs_idx      = thrust::distance(local_point_offsets, local_point_offset);
    uint32_t const point_idx      = quad_point_offsets[pairs_idx] + (i - *local_point_offset);
    uint32_t const poly_idx       = poly_indices.element<uint32_t>(pairs_idx);
    auto const &point             = points[point_idx];
    auto const distance           = point_to_poly_line_distance(thrust::get<0>(point),
                                                      thrust::get<1>(point),
                                                      poly_idx,
                                                      poly_offsets,
                                                      poly_points_x,
                                                      poly_points_y);
    return thrust::make_tuple(point_idx, poly_idx, distance);
  }
};

struct compute_quadtree_point_to_nearest_polyline {
  template <typename T, typename... Args>
  std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<cudf::table>> operator()(
    Args &&...)
  {
    CUDF_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::table>> operator()(
    cudf::table_view const &poly_quad_pairs,
    cudf::table_view const &quadtree,
    cudf::column_view const &point_indices,
    cudf::column_view const &point_x,
    cudf::column_view const &point_y,
    cudf::column_view const &poly_offsets,
    cudf::column_view const &poly_points_x,
    cudf::column_view const &poly_points_y,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr)
  {
    // Wrapped in an IIFE so `local_point_offsets` is freed on return
    auto poly_point_idxs_and_distances = [&]() {
      auto quad_lengths        = quadtree.column(3);
      auto quad_offsets        = quadtree.column(4);
      auto poly_indices        = poly_quad_pairs.column(0);
      auto quad_indices        = poly_quad_pairs.column(1);
      auto num_poly_quad_pairs = poly_quad_pairs.num_rows();

      auto counting_iter     = thrust::make_counting_iterator(0);
      auto quad_lengths_iter = thrust::make_permutation_iterator(quad_lengths.begin<uint32_t>(),
                                                                 quad_indices.begin<uint32_t>());

      auto quad_offsets_iter = thrust::make_permutation_iterator(quad_offsets.begin<uint32_t>(),
                                                                 quad_indices.begin<uint32_t>());

      // Compute a "local" set of zero-based point offsets from number of points in each quadrant
      // Use `num_poly_quad_pairs + 1` as the length so that the last element produced by
      // `inclusive_scan` is the total number of points to be tested against any polyline.
      rmm::device_uvector<uint32_t> local_point_offsets(num_poly_quad_pairs + 1, stream);

      thrust::inclusive_scan(rmm::exec_policy(stream),
                             quad_lengths_iter,
                             quad_lengths_iter + num_poly_quad_pairs,
                             local_point_offsets.begin() + 1);

      // Ensure local point offsets starts at 0
      uint32_t init{0};
      local_point_offsets.set_element_async(0, init, stream);

      // The last element is the total number of points to test against any polyline.
      auto num_total_points = local_point_offsets.back_element(stream);

      // Allocate memory for the polyline/point indices and distances
      rmm::device_uvector<uint32_t> poly_idxs(num_total_points, stream);
      rmm::device_uvector<uint32_t> point_idxs(num_total_points, stream);
      rmm::device_uvector<T> poly_point_distances(num_total_points, stream);

      auto point_poly_indices_and_distances =
        make_zip_iterator(point_idxs.begin(), poly_idxs.begin(), poly_point_distances.begin());

      // Enumerate the point X/Ys using the sorted `point_indices` (from quadtree construction)
      auto point_xys_iter = thrust::make_permutation_iterator(
        make_zip_iterator(point_x.begin<T>(), point_y.begin<T>()), point_indices.begin<uint32_t>());

      // Compute the combination of point and polyline index pairs. For each polyline/quadrant pair,
      // enumerate pairs of (point_index, poly_index) for each point in each quadrant, and calculate
      // the minimum distance between each point/poly pair.
      //
      // In Python pseudocode:
      // ```
      // pp_pairs_and_dist = []
      // for polyline, quadrant in pq_pairs:
      //   for point in quadrant:
      //     pp_pairs_and_dist.append((point, polyline, min_distance(point, polyline)))
      // ```
      //
      thrust::transform(rmm::exec_policy(stream),
                        counting_iter,
                        counting_iter + num_total_points,
                        point_poly_indices_and_distances,
                        compute_point_poly_indices_and_distances<T,
                                                                 decltype(point_xys_iter),
                                                                 decltype(quad_offsets_iter)>{
                          point_xys_iter,
                          quad_offsets_iter,
                          local_point_offsets.begin(),
                          local_point_offsets.size() - 1,
                          *cudf::column_device_view::create(poly_indices, stream),
                          *cudf::column_device_view::create(poly_offsets, stream),
                          *cudf::column_device_view::create(poly_points_x, stream),
                          *cudf::column_device_view::create(poly_points_y, stream)});

      // sort the point/polyline indices and distances for `reduce_by_key` below
      thrust::sort_by_key(rmm::exec_policy(stream),
                          point_idxs.begin(),
                          point_idxs.end(),
                          point_poly_indices_and_distances);

      return std::make_tuple(
        std::move(poly_idxs), std::move(point_idxs), std::move(poly_point_distances));
    }();

    auto &poly_idxs            = std::get<0>(poly_point_idxs_and_distances);
    auto &point_idxs           = std::get<1>(poly_point_idxs_and_distances);
    auto &poly_point_distances = std::get<2>(poly_point_idxs_and_distances);

    // Allocate output columns for the point and polyline index pairs and their distances
    auto point_index_col = make_fixed_width_column<uint32_t>(point_x.size(), stream, mr);
    auto poly_index_col  = make_fixed_width_column<uint32_t>(point_x.size(), stream, mr);
    auto distance_col    = make_fixed_width_column<T>(point_x.size(), stream, mr);

    // Fill output distance column with T_MAX because `reduce_by_key` selector is associative
    thrust::fill(rmm::exec_policy(stream),
                 distance_col->mutable_view().template begin<T>(),
                 distance_col->mutable_view().template end<T>(),
                 std::numeric_limits<T>::max());

    // Reduce the intermediate point/poly indices to lists of point/polyline
    // index pairs and distances, selecting the polyline index closest for each point.
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          // keys_first
                          point_idxs.begin(),
                          // keys_last
                          point_idxs.end(),
                          // values_first
                          make_zip_iterator(poly_idxs.begin(), poly_point_distances.begin()),
                          // keys_output
                          point_index_col->mutable_view().begin<uint32_t>(),
                          // values_output
                          make_zip_iterator(poly_index_col->mutable_view().begin<uint32_t>(),
                                            distance_col->mutable_view().template begin<T>()),
                          // binary_pred
                          thrust::equal_to<uint32_t>(),
                          // binary_op
                          [] __device__(auto const &a, auto const &b) {
                            return thrust::get<1>(a) < thrust::get<1>(b) ? a : b;
                          });

    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(3);
    cols.push_back(std::move(point_index_col));
    cols.push_back(std::move(poly_index_col));
    cols.push_back(std::move(distance_col));
    return std::make_unique<cudf::table>(std::move(cols));
  }
};

}  // namespace

std::unique_ptr<cudf::table> quadtree_point_to_nearest_polyline(
  cudf::table_view const &poly_quad_pairs,
  cudf::table_view const &quadtree,
  cudf::column_view const &point_indices,
  cudf::column_view const &point_x,
  cudf::column_view const &point_y,
  cudf::column_view const &poly_offsets,
  cudf::column_view const &poly_points_x,
  cudf::column_view const &poly_points_y,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr)
{
  return cudf::type_dispatcher(point_x.type(),
                               compute_quadtree_point_to_nearest_polyline{},
                               poly_quad_pairs,
                               quadtree,
                               point_indices,
                               point_x,
                               point_y,
                               poly_offsets,
                               poly_points_x,
                               poly_points_y,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> quadtree_point_to_nearest_polyline(
  cudf::table_view const &poly_quad_pairs,
  cudf::table_view const &quadtree,
  cudf::column_view const &point_indices,
  cudf::column_view const &point_x,
  cudf::column_view const &point_y,
  cudf::column_view const &poly_offsets,
  cudf::column_view const &poly_points_x,
  cudf::column_view const &poly_points_y,
  rmm::mr::device_memory_resource *mr)
{
  CUSPATIAL_EXPECTS(poly_quad_pairs.num_columns() == 2,
                    "a quadrant-polyline table must have 2 columns");
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "a quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(point_indices.size() == point_x.size() && point_x.size() == point_y.size(),
                    "number of points must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(poly_points_x.size() == poly_points_y.size(),
                    "numbers of vertices must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(poly_points_x.size() >= 2 * poly_offsets.size(),
                    "all polylines must have at least two vertices");
  CUSPATIAL_EXPECTS(poly_points_x.type() == poly_points_y.type(),
                    "polyline columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == point_y.type(), "point columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == poly_points_x.type(),
                    "points and polylines must have the same data type");

  if (poly_quad_pairs.num_rows() == 0 || quadtree.num_rows() == 0 || point_indices.size() == 0 ||
      poly_offsets.size() == 0) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(3);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    cols.push_back(cudf::make_empty_column(point_x.type()));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::quadtree_point_to_nearest_polyline(poly_quad_pairs,
                                                    quadtree,
                                                    point_indices,
                                                    point_x,
                                                    point_y,
                                                    poly_offsets,
                                                    poly_points_x,
                                                    poly_points_y,
                                                    rmm::cuda_stream_default,
                                                    mr);
}

}  // namespace cuspatial
