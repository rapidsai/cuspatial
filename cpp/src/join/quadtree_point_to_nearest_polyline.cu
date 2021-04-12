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
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <memory>

namespace cuspatial {
namespace detail {
namespace {

inline __device__ thrust::tuple<uint32_t, uint32_t> get_quad_poly_and_local_point_indices(
  uint32_t const global_index, uint32_t const *point_offsets, uint32_t const *point_offsets_end)
{
  // Calculate the position in "point_offsets" that `global_index` falls between.
  // This position is the index of the poly/quad pair for this `global_index`.
  //
  // Dereferencing `local_point_offset` yields the zero-based first point position of this
  // quadrant. Adding this zero-based position to the quadrant's first point position in the
  // quadtree yields the "global" position in the `point_indices` map.
  auto const local_point_offset =
    thrust::upper_bound(thrust::seq, point_offsets, point_offsets_end, global_index) - 1;
  return thrust::make_tuple(
    // quad_poly_index
    thrust::distance(point_offsets, local_point_offset),
    // local_point_index
    global_index - *local_point_offset);
}

inline __device__ thrust::tuple<uint32_t, uint32_t> get_local_poly_index_and_count(
  uint32_t const poly_index, uint32_t const *quad_offsets, uint32_t const *quad_offsets_end)
{
  auto const lhs_end     = quad_offsets;
  auto const rhs_end     = quad_offsets_end;
  auto const quad_offset = quad_offsets[poly_index];
  auto const lhs =
    thrust::lower_bound(thrust::seq, lhs_end, quad_offsets + poly_index, quad_offset);
  auto const rhs =
    thrust::upper_bound(thrust::seq, quad_offsets + poly_index, rhs_end, quad_offset);

  return thrust::make_tuple(
    // local_poly_index
    static_cast<uint32_t>(thrust::distance(lhs, quad_offsets + poly_index)),
    // num_polys_in_quad
    static_cast<uint32_t>(thrust::distance(lhs, rhs)));
}

inline __device__ thrust::tuple<uint32_t, uint32_t> get_transposed_point_and_pair_index(
  uint32_t const global_index,
  uint32_t const *point_offsets,
  uint32_t const *point_offsets_end,
  uint32_t const *quad_offsets,
  uint32_t const *quad_offsets_end,
  uint32_t const *quad_lengths)
{
  uint32_t quad_poly_index{0}, local_point_offset{0};
  thrust::tie(quad_poly_index, local_point_offset) =
    get_quad_poly_and_local_point_indices(global_index, point_offsets, point_offsets_end);

  uint32_t local_poly_index{0}, num_polys_in_quad{0};
  thrust::tie(local_poly_index, num_polys_in_quad) =
    get_local_poly_index_and_count(quad_poly_index, quad_offsets, quad_offsets_end);

  auto const quad_point_offset      = quad_offsets[quad_poly_index];
  auto const num_points_in_quad     = quad_lengths[quad_poly_index];
  auto const quad_poly_offset       = quad_poly_index - local_poly_index;
  auto const quad_poly_point_start  = local_poly_index * num_points_in_quad;
  auto const transposed_point_start = quad_poly_point_start + local_point_offset;

  return thrust::make_tuple(
    // transposed point index
    (transposed_point_start / num_polys_in_quad) + quad_point_offset,
    // transposed polyline index
    (transposed_point_start % num_polys_in_quad) + quad_poly_offset);
}

template <typename T, typename PointIter>
struct compute_point_poly_indices_and_distances {
  PointIter points;
  uint32_t const *point_offsets;
  uint32_t const *point_offsets_end;
  uint32_t const *quad_offsets;
  uint32_t const *quad_offsets_end;
  uint32_t const *quad_lengths;
  uint32_t const *poly_indices;
  cudf::column_device_view const poly_offsets;
  cudf::column_device_view const poly_points_x;
  cudf::column_device_view const poly_points_y;
  inline __device__ thrust::tuple<uint32_t, uint32_t, T> operator()(uint32_t const global_index)
  {
    // (point_index, poly_index)
    auto const ids = get_transposed_point_and_pair_index(
      global_index, point_offsets, point_offsets_end, quad_offsets, quad_offsets_end, quad_lengths);

    auto const point_id = thrust::get<0>(ids);
    auto const point_xy = points[point_id];
    auto const poly_idx = poly_indices[thrust::get<1>(ids)];
    auto const distance = point_to_poly_line_distance<T>(thrust::get<0>(point_xy),  // x
                                                         thrust::get<1>(point_xy),  // y
                                                         poly_idx,
                                                         poly_offsets,
                                                         poly_points_x,
                                                         poly_points_y);
    return thrust::make_tuple(point_id, poly_idx, distance);
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
    // Wrapped in an IIFE so intermediate allocations are freed on return
    auto reduction = [&]() {
      auto quad_lengths        = quadtree.column(3);
      auto quad_offsets        = quadtree.column(4);
      auto num_poly_quad_pairs = poly_quad_pairs.num_rows();

      rmm::device_uvector<uint32_t> quad_poly_indices(num_poly_quad_pairs, stream);
      rmm::device_uvector<uint32_t> quad_point_offsets(num_poly_quad_pairs, stream);
      rmm::device_uvector<uint32_t> quad_point_lengths(num_poly_quad_pairs, stream);

      auto quad_poly_indices_and_quad_offsets_lengths = thrust::make_zip_iterator(
        // poly_indices
        poly_quad_pairs.column(0).begin<uint32_t>(),
        // quad_offsets
        thrust::make_permutation_iterator(quadtree.column(4).begin<uint32_t>(),
                                          poly_quad_pairs.column(1).begin<uint32_t>()),
        // quad_lengths
        thrust::make_permutation_iterator(quadtree.column(3).begin<uint32_t>(),
                                          poly_quad_pairs.column(1).begin<uint32_t>()));

      // Copy the quad/polyline indices into local vectors instead of sorting the input columns
      thrust::copy(
        rmm::exec_policy(stream),
        quad_poly_indices_and_quad_offsets_lengths,
        quad_poly_indices_and_quad_offsets_lengths + num_poly_quad_pairs,
        thrust::make_zip_iterator(
          quad_poly_indices.begin(), quad_point_offsets.begin(), quad_point_lengths.begin()));

      // Sort the quad/polyline index pairs by quadrant so that the `point_indices` are
      // produced/reduced in ascending order below.
      thrust::stable_sort_by_key(
        rmm::exec_policy(stream),
        quad_point_offsets.begin(),
        quad_point_offsets.end(),
        thrust::make_zip_iterator(quad_point_lengths.begin(), quad_poly_indices.begin()));

      // Compute a "local" set of zero-based point offsets from number of points in each quadrant
      // Use `num_poly_quad_pairs + 1` as the length so that the last element produced by
      // `inclusive_scan` is the total number of points to be tested against any polyline.
      rmm::device_uvector<uint32_t> local_point_offsets(num_poly_quad_pairs + 1, stream);

      thrust::inclusive_scan(rmm::exec_policy(stream),
                             quad_point_lengths.begin(),
                             quad_point_lengths.end(),
                             local_point_offsets.begin() + 1);

      // Ensure local point offsets starts at 0
      uint32_t init{0};
      local_point_offsets.set_element_async(0, init, stream);

      // The last element is the total number of points to test against any polyline.
      auto num_point_poly_pairs = local_point_offsets.back_element(stream);

      // Enumerate the point X/Ys using the sorted `point_indices` (from quadtree construction)
      auto point_xys_iter = thrust::make_permutation_iterator(
        thrust::make_zip_iterator(point_x.begin<T>(), point_y.begin<T>()),
        point_indices.begin<uint32_t>());

      //
      // Compute the combination of point and polyline index pairs. For each polyline/quadrant pair,
      // enumerate pairs of (point_index, polyline_index) for each point in each quadrant, and
      // calculate the minimum distance between each point/poly pair.
      //
      // In Python pseudocode:
      // ```
      // pp_pairs_and_dist = []
      // for polyline, quadrant in pq_pairs:
      //   for point in quadrant:
      //     pp_pairs_and_dist.append((point, polyline, min_distance(point, polyline)))
      // ```
      //
      // However, the above psuedocode produces values in an order such that the distance
      // from a point to each polyline cannot be reduced with `thrust::reduce_by_key`:
      // ```
      //   point | polyline | distance
      //       0 |        0 |     10.0
      //       1 |        0 |     30.0
      //       2 |        0 |     20.0
      //       0 |        1 |     30.0
      //       1 |        1 |     20.0
      //       2 |        1 |     10.0
      // ```
      //
      // In order to use `thrust::reduce_by_key` to compute the minimum distance from a point to
      // the polylines in its quadrant, the above table needs to be sorted by `point` instead of
      // `polyline`:
      // ```
      //   point | polyline | distance
      //       0 |        0 |     10.0
      //       0 |        1 |     30.0
      //       1 |        0 |     30.0
      //       1 |        1 |     20.0
      //       2 |        0 |     20.0
      //       2 |        1 |     10.0
      // ```
      //
      // A naive approach would be to allocate memory for the above three columns, sort the
      // columns by `point`, then use `thrust::reduce_by_key` to compute the min distances.
      //
      // The sizes of the intermediate buffers required can easily grow beyond available
      // device memory, so a better approach is to use a Thrust iterator to yield values
      // in the sorted order on demand instead, which is what we're doing here.
      //

      auto all_point_poly_indices_and_distances = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0u),
        compute_point_poly_indices_and_distances<T, decltype(point_xys_iter)>{
          point_xys_iter,
          local_point_offsets.begin(),
          local_point_offsets.end(),
          quad_point_offsets.begin(),
          quad_point_offsets.end(),
          quad_point_lengths.begin(),
          quad_poly_indices.begin(),
          *cudf::column_device_view::create(poly_offsets, stream),
          *cudf::column_device_view::create(poly_points_x, stream),
          *cudf::column_device_view::create(poly_points_y, stream)});

      auto all_point_indices =
        thrust::make_transform_iterator(all_point_poly_indices_and_distances,
                                        [] __device__(auto const &x) { return thrust::get<0>(x); });

      // Allocate vectors for the distances min reduction
      rmm::device_uvector<uint32_t> point_idxs(point_x.size(), stream);
      rmm::device_uvector<uint32_t> poly_idxs(point_x.size(), stream);
      rmm::device_uvector<T> distances(point_x.size(), stream);

      // Reduce the intermediate point/polyline indices to lists of point/polyline index pairs and
      // distances, selecting the polyline index closest to each point.
      auto const num_distances = thrust::distance(
        point_idxs.begin(),
        thrust::reduce_by_key(
          rmm::exec_policy(stream),
          // point indices in
          all_point_indices,
          all_point_indices + num_point_poly_pairs,
          all_point_poly_indices_and_distances,
          // point indices out
          point_idxs.begin(),
          // point/polyline indices and distances out
          thrust::make_zip_iterator(
            thrust::make_discard_iterator(), poly_idxs.begin(), distances.begin()),
          // comparator
          thrust::equal_to<uint32_t>(),
          // binop to select the point/polyline pair with the smallest distance
          [] __device__(auto const &lhs, auto const &rhs) {
            T const &d_lhs = thrust::get<2>(lhs);
            T const &d_rhs = thrust::get<2>(rhs);
            return (d_lhs == T{0}) ? rhs : (d_rhs == T{0}) ? lhs : d_lhs < d_rhs ? lhs : rhs;
          })
          .first);

      point_idxs.resize(num_distances, stream);
      point_idxs.shrink_to_fit(stream);

      poly_idxs.resize(num_distances, stream);
      poly_idxs.shrink_to_fit(stream);

      distances.resize(num_distances, stream);
      distances.shrink_to_fit(stream);

      return std::make_tuple(std::move(point_idxs), std::move(poly_idxs), std::move(distances));
    }();

    auto const &point_idxs = std::get<0>(reduction);
    auto const &poly_idxs  = std::get<1>(reduction);
    auto const &distances  = std::get<2>(reduction);

    // Allocate output columns for the point and polyline index pairs and their distances
    auto point_index_col = make_fixed_width_column<uint32_t>(point_x.size(), stream, mr);
    auto poly_index_col  = make_fixed_width_column<uint32_t>(point_x.size(), stream, mr);
    auto distance_col    = make_fixed_width_column<T>(point_x.size(), stream, mr);

    // scatter the values from their positions after reduction into their output positions
    thrust::scatter(
      rmm::exec_policy(stream),
      thrust::make_zip_iterator(point_idxs.begin(), poly_idxs.begin(), distances.begin()),
      thrust::make_zip_iterator(point_idxs.end(), poly_idxs.end(), distances.end()),
      point_idxs.begin(),
      thrust::make_zip_iterator(point_index_col->mutable_view().begin<uint32_t>(),
                                poly_index_col->mutable_view().begin<uint32_t>(),
                                distance_col->mutable_view().template begin<T>()));

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
