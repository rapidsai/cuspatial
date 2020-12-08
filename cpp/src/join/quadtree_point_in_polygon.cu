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
#include <utility/point_in_polygon.cuh>

#include <cuspatial/error.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform.h>

namespace cuspatial {
namespace detail {
namespace {

template <typename T, typename QuadOffsetsIter>
struct compute_poly_and_point_indices {
  QuadOffsetsIter quad_point_offsets;
  uint32_t const *local_point_offsets;
  size_t const num_local_point_offsets;
  cudf::column_device_view const poly_indices;
  thrust::tuple<uint32_t, uint32_t> __device__ operator()(cudf::size_type const i)
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
    return thrust::make_tuple(poly_idx, point_idx);
  }
};

template <typename T, typename PointIter>
struct test_poly_point_intersection {
  PointIter points;
  cudf::column_device_view const poly_offsets;
  cudf::column_device_view const ring_offsets;
  cudf::column_device_view const poly_points_x;
  cudf::column_device_view const poly_points_y;

  bool __device__ operator()(thrust::tuple<uint32_t, uint32_t> const &poly_point_idxs)
  {
    auto &poly_idx  = thrust::get<0>(poly_point_idxs);
    auto &point_idx = thrust::get<1>(poly_point_idxs);
    auto point      = points[point_idx];
    return not is_point_in_polygon(thrust::get<0>(point),
                                   thrust::get<1>(point),
                                   poly_idx,
                                   poly_offsets,
                                   ring_offsets,
                                   poly_points_x,
                                   poly_points_y);
  }
};

struct compute_quadtree_point_in_polygon {
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
    cudf::column_view const &ring_offsets,
    cudf::column_view const &poly_points_x,
    cudf::column_view const &poly_points_y,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr)
  {
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
    // `inclusive_scan` is the total number of points to be tested against any polygon.
    rmm::device_uvector<uint32_t> local_point_offsets(num_poly_quad_pairs + 1, stream);

    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream.value()),
                           quad_lengths_iter,
                           quad_lengths_iter + num_poly_quad_pairs,
                           local_point_offsets.begin() + 1);

    // Ensure local point offsets starts at 0
    local_point_offsets.set_element_async(0, 0, stream);

    // The last element is the total number of points to test against any polygon.
    auto num_total_points = local_point_offsets.back_element(stream);

    // Allocate memory for the polygon and point index pairs
    rmm::device_uvector<uint32_t> poly_idxs(num_total_points, stream);
    rmm::device_uvector<uint32_t> point_idxs(num_total_points, stream);

    auto poly_and_point_indices = make_zip_iterator(poly_idxs.begin(), point_idxs.begin());

    // Compute the combination of polygon and point index pairs. For each polygon/quadrant pair,
    // enumerate pairs of (poly_index, point_index) for each point in each quadrant.
    //
    // In Python pseudocode:
    // ```
    // pp_pairs = []
    // for polygon, quadrant in pq_pairs:
    //   for point in quadrant:
    //     pp_pairs.append((polygon, point))
    // ```
    //
    thrust::transform(rmm::exec_policy(stream)->on(stream.value()),
                      counting_iter,
                      counting_iter + num_total_points,
                      poly_and_point_indices,
                      compute_poly_and_point_indices<T, decltype(quad_offsets_iter)>{
                        quad_offsets_iter,
                        local_point_offsets.begin(),
                        local_point_offsets.size() - 1,
                        *cudf::column_device_view::create(poly_indices, stream)});

    // Enumerate the point X/Ys using the sorted `point_indices` (from quadtree construction)
    auto point_xys_iter = thrust::make_permutation_iterator(
      make_zip_iterator(point_x.begin<T>(), point_y.begin<T>()), point_indices.begin<uint32_t>());

    // Compute the number of intersections by removing (poly, point) pairs that don't intersect
    auto num_intersections = thrust::distance(
      poly_and_point_indices,
      thrust::remove_if(rmm::exec_policy(stream)->on(stream.value()),
                        poly_and_point_indices,
                        poly_and_point_indices + num_total_points,
                        test_poly_point_intersection<T, decltype(point_xys_iter)>{
                          point_xys_iter,
                          *cudf::column_device_view::create(poly_offsets, stream),
                          *cudf::column_device_view::create(ring_offsets, stream),
                          *cudf::column_device_view::create(poly_points_x, stream),
                          *cudf::column_device_view::create(poly_points_y, stream)}));

    // Allocate output columns for the number of pairs that intersected
    auto poly_idx_col  = make_fixed_width_column<uint32_t>(num_intersections, stream, mr);
    auto point_idx_col = make_fixed_width_column<uint32_t>(num_intersections, stream, mr);

    // Note: no need to resize `poly_idxs` or `point_idxs` if we set the end iterator to
    // `idxs.begin() + num_intersections`.

    // populate the polygon indices column
    thrust::copy(rmm::exec_policy(stream)->on(stream.value()),
                 poly_idxs.begin(),
                 poly_idxs.begin() + num_intersections,
                 poly_idx_col->mutable_view().template begin<uint32_t>());

    // populate the point indices column
    thrust::copy(rmm::exec_policy(stream)->on(stream.value()),
                 point_idxs.begin(),
                 point_idxs.begin() + num_intersections,
                 point_idx_col->mutable_view().template begin<uint32_t>());

    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(std::move(poly_idx_col));
    cols.push_back(std::move(point_idx_col));
    return std::make_unique<cudf::table>(std::move(cols));
  }
};

}  // namespace

std::unique_ptr<cudf::table> quadtree_point_in_polygon(cudf::table_view const &poly_quad_pairs,
                                                       cudf::table_view const &quadtree,
                                                       cudf::column_view const &point_indices,
                                                       cudf::column_view const &point_x,
                                                       cudf::column_view const &point_y,
                                                       cudf::column_view const &poly_offsets,
                                                       cudf::column_view const &ring_offsets,
                                                       cudf::column_view const &poly_points_x,
                                                       cudf::column_view const &poly_points_y,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource *mr)
{
  return cudf::type_dispatcher(point_x.type(),
                               compute_quadtree_point_in_polygon{},
                               poly_quad_pairs,
                               quadtree,
                               point_indices,
                               point_x,
                               point_y,
                               poly_offsets,
                               ring_offsets,
                               poly_points_x,
                               poly_points_y,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> quadtree_point_in_polygon(cudf::table_view const &poly_quad_pairs,
                                                       cudf::table_view const &quadtree,
                                                       cudf::column_view const &point_indices,
                                                       cudf::column_view const &point_x,
                                                       cudf::column_view const &point_y,
                                                       cudf::column_view const &poly_offsets,
                                                       cudf::column_view const &ring_offsets,
                                                       cudf::column_view const &poly_points_x,
                                                       cudf::column_view const &poly_points_y,
                                                       rmm::mr::device_memory_resource *mr)
{
  CUSPATIAL_EXPECTS(poly_quad_pairs.num_columns() == 2,
                    "a quadrant-polygon table must have 2 columns");
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "a quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(point_indices.size() == point_x.size() && point_x.size() == point_y.size(),
                    "number of points must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(ring_offsets.size() >= poly_offsets.size(),
                    "number of rings must be no less than number of polygons");
  CUSPATIAL_EXPECTS(poly_points_x.size() == poly_points_y.size(),
                    "numbers of vertices must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(poly_points_x.size() >= 3 * ring_offsets.size(),
                    "all rings must have at least 3 vertices");
  CUSPATIAL_EXPECTS(poly_points_x.type() == poly_points_y.type(),
                    "polygon columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == point_y.type(), "point columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == poly_points_x.type(),
                    "points and polygons must have the same data type");

  if (poly_quad_pairs.num_rows() == 0 || quadtree.num_rows() == 0 || point_indices.size() == 0 ||
      poly_offsets.size() == 0) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::quadtree_point_in_polygon(poly_quad_pairs,
                                           quadtree,
                                           point_indices,
                                           point_x,
                                           point_y,
                                           poly_offsets,
                                           ring_offsets,
                                           poly_points_x,
                                           poly_points_y,
                                           rmm::cuda_stream_default,
                                           mr);
}

}  // namespace cuspatial
