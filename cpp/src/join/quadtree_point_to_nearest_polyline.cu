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

#include "indexing/construction/detail/utilities.cuh"

#include <cuspatial/error.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <vector>

namespace cuspatial {
namespace detail {
namespace {

static uint32_t const threads_per_block = 256;

template <typename T>
__global__ void find_nearest_polyline_kernel(
  uint32_t const *quad_idxs,            // point quadrant id array -base 0
  uint32_t const num_poly_idx_offsets,  // number of polyline index offsets
  uint32_t const *poly_idx_offsets,     // starting positions of the first polyline idx
  uint32_t const num_poly_idxs,         // number of polyline indices
  uint32_t const *poly_idxs,

  cudf::column_device_view quad_lengths,  // numbers of points in each quadrant
  cudf::column_device_view quad_offsets,  // offset of first point in each quadrant
  cudf::column_device_view point_indices,
  cudf::column_device_view point_x,
  cudf::column_device_view point_y,

  cudf::column_device_view poly_offsets,  // positions of the first vertex in a polyline
  cudf::column_device_view poly_points_x,
  cudf::column_device_view poly_points_y,

  cudf::mutable_column_device_view out_point_index,
  cudf::mutable_column_device_view out_poly_index,
  cudf::mutable_column_device_view out_distance)
{
  // each block processes a quadrant
  auto quad_pos = blockIdx.x + gridDim.x * blockIdx.y;
  auto quad_idx = quad_idxs[quad_pos];

  auto poly_begin = poly_idx_offsets[quad_pos];
  auto poly_end =
    quad_pos < num_poly_idx_offsets - 1 ? poly_idx_offsets[quad_pos + 1] : num_poly_idxs;

  auto num_points = quad_lengths.element<uint32_t>(quad_idx);

  for (auto tid = threadIdx.x; tid < num_points; tid += blockDim.x) {
    // each thread loads its point
    auto point_pos       = quad_offsets.element<uint32_t>(quad_idx) + tid;
    auto point_id        = point_indices.element<uint32_t>(point_pos);
    auto x               = point_x.element<T>(point_id);
    auto y               = point_y.element<T>(point_id);
    T distance           = 1e20;
    auto nearest_poly_id = static_cast<uint32_t>(-1);

    for (uint32_t poly_id = poly_begin; poly_id < poly_end; poly_id++)  // for each polyline
    {
      auto poly_idx   = poly_idxs[poly_id];
      auto ring_begin = poly_offsets.element<uint32_t>(poly_idx);
      auto ring_end   = poly_idx < poly_offsets.size() - 1
                        ? poly_offsets.element<uint32_t>(poly_idx + 1)
                        : poly_points_x.size();
      auto ring_len = ring_end - ring_begin;

      for (auto point_idx = 0; point_idx < ring_len; point_idx++)  // for each line
      {
        T x0  = poly_points_x.element<T>(ring_begin + ((point_idx + 0) % ring_len));
        T y0  = poly_points_y.element<T>(ring_begin + ((point_idx + 0) % ring_len));
        T x1  = poly_points_x.element<T>(ring_begin + ((point_idx + 1) % ring_len));
        T y1  = poly_points_y.element<T>(ring_begin + ((point_idx + 1) % ring_len));
        T dx  = x1 - x0;
        T dy  = y1 - y0;
        T dx2 = x - x0;
        T dy2 = y - y0;
        T r   = (dx * dx2 + dy * dy2) / sqrt(dx * dx + dy * dy);
        T d   = 1e20;
        if (r <= 0 || r >= sqrt(dx * dx + dy * dy)) {
          T d1 = hypot(x - x0, y - y0);
          T d2 = hypot(x - x1, y - y1);
          d    = min(min(d, d1), d2);
        } else {
          d = sqrt((dx2 * dx2 + dy2 * dy2) - (r * r));
        }
        if (d < distance) {
          distance        = d;
          nearest_poly_id = poly_idx;
        }
      }
    }

    // TODO: use input point id
    out_point_index.element<uint32_t>(point_pos) = point_pos;
    out_poly_index.element<uint32_t>(point_pos)  = nearest_poly_id;
    out_distance.element<T>(point_pos)           = distance;
  }
}

template <typename T>
std::unique_ptr<cudf::table> compute_quadtree_point_to_nearest_polyline(
  cudf::column_view const &poly_idx,
  cudf::column_view const &quad_idx,
  cudf::column_view const &quad_lengths,
  cudf::column_view const &quad_offsets,
  cudf::column_view const &point_indices,
  cudf::column_view const &point_x,
  cudf::column_view const &point_y,
  cudf::column_view const &poly_offsets,
  cudf::column_view const &poly_points_x,
  cudf::column_view const &poly_points_y,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream)
{
  const uint32_t num_pairs = poly_idx.size();

  rmm::device_uvector<uint32_t> d_poly_idx(num_pairs, stream);

  thrust::copy(rmm::exec_policy(stream)->on(stream),
               poly_idx.begin<uint32_t>(),
               poly_idx.end<uint32_t>(),
               d_poly_idx.begin());

  rmm::device_uvector<uint32_t> d_quad_idx(num_pairs, stream);

  thrust::copy(rmm::exec_policy(stream)->on(stream),
               quad_idx.begin<uint32_t>(),
               quad_idx.end<uint32_t>(),
               d_quad_idx.begin());

  // sort (d_poly_idx, d_quad_idx) using d_quad_idx as key => (quad_idxs, poly_idxs)
  thrust::sort_by_key(
    rmm::exec_policy(stream)->on(stream), d_quad_idx.begin(), d_quad_idx.end(), d_poly_idx.begin());

  // reduce_by_key using d_quad_idx as the key
  // exclusive_scan on numbers of polygons associated with a quadrant to create d_poly_idx_offsets

  rmm::device_uvector<uint32_t> d_poly_idx_offsets(num_pairs, stream);

  uint32_t num_quads =
    thrust::distance(d_poly_idx_offsets.begin(),
                     thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                                           d_quad_idx.begin(),
                                           d_quad_idx.end(),
                                           thrust::constant_iterator<uint32_t>(1),
                                           d_quad_idx.begin(),
                                           d_poly_idx_offsets.begin())
                       .second);

  d_quad_idx.resize(num_quads, stream);
  d_poly_idx_offsets.resize(num_quads, stream);

  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         d_poly_idx_offsets.begin(),
                         d_poly_idx_offsets.end(),
                         d_poly_idx_offsets.begin());

  auto point_index_col   = make_fixed_width_column<uint32_t>(point_x.size(), stream, mr);
  auto poly_index_col    = make_fixed_width_column<uint32_t>(point_x.size(), stream, mr);
  auto poly_distance_col = make_fixed_width_column<T>(point_x.size(), stream, mr);

  find_nearest_polyline_kernel<T><<<num_quads, threads_per_block, 0, stream>>>(
    d_quad_idx.begin(),
    d_poly_idx_offsets.size(),
    d_poly_idx_offsets.begin(),
    d_poly_idx.size(),
    d_poly_idx.begin(),
    *cudf::column_device_view::create(quad_lengths, stream),
    *cudf::column_device_view::create(quad_offsets, stream),
    *cudf::column_device_view::create(point_indices, stream),
    *cudf::column_device_view::create(point_x, stream),
    *cudf::column_device_view::create(point_y, stream),
    *cudf::column_device_view::create(poly_offsets, stream),
    *cudf::column_device_view::create(poly_points_x, stream),
    *cudf::column_device_view::create(poly_points_y, stream),
    *cudf::mutable_column_device_view::create(*point_index_col, stream),
    *cudf::mutable_column_device_view::create(*poly_index_col, stream),
    *cudf::mutable_column_device_view::create(*poly_distance_col, stream));

  CUDA_TRY(cudaStreamSynchronize(stream));

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(3);
  cols.push_back(std::move(point_index_col));
  cols.push_back(std::move(poly_index_col));
  cols.push_back(std::move(poly_distance_col));
  return std::make_unique<cudf::table>(std::move(cols));
}

struct dispatch_quadtree_point_to_nearest_polyline {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  std::unique_ptr<cudf::table> operator()(cudf::table_view const &poly_quad_pairs,
                                          cudf::table_view const &quadtree,
                                          cudf::column_view const &point_indices,
                                          cudf::column_view const &point_x,
                                          cudf::column_view const &point_y,
                                          cudf::column_view const &poly_offsets,
                                          cudf::column_view const &poly_points_x,
                                          cudf::column_view const &poly_points_y,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
  {
    return compute_quadtree_point_to_nearest_polyline<T>(poly_quad_pairs.column(0),
                                                         poly_quad_pairs.column(1),
                                                         quadtree.column(3),
                                                         quadtree.column(4),
                                                         point_indices,
                                                         point_x,
                                                         point_y,
                                                         poly_offsets,
                                                         poly_points_x,
                                                         poly_points_y,
                                                         mr,
                                                         stream);
  }

  template <typename T,
            std::enable_if_t<!std::is_floating_point<T>::value> * = nullptr,
            typename... Args>
  std::unique_ptr<cudf::table> operator()(Args &&...)
  {
    CUDF_FAIL("Non-floating point operation is not supported");
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
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream)
{
  return cudf::type_dispatcher(point_x.type(),
                               dispatch_quadtree_point_to_nearest_polyline{},
                               poly_quad_pairs,
                               quadtree,
                               point_indices,
                               point_x,
                               point_y,
                               poly_offsets,
                               poly_points_x,
                               poly_points_y,
                               mr,
                               stream);
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
                    "a quadrant-polygon table must have 2 columns");
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "a quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(point_indices.size() == point_x.size() && point_x.size() == point_y.size(),
                    "number of points must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(poly_points_x.size() == poly_points_y.size(),
                    "numbers of vertices must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(poly_points_x.size() >= 2 * poly_offsets.size(),
                    "all polylines must have at least two vertices");
  CUSPATIAL_EXPECTS(poly_points_x.type() == poly_points_y.type(),
                    "polygon columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == point_y.type(), "point columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == poly_points_x.type(),
                    "points and polygons must have the same data type");

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
                                                    mr,
                                                    cudaStream_t{0});
}

}  // namespace cuspatial
