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

static uint8_t const max_warps_per_block  = 32;
static uint8_t const num_threads_per_warp = 32;
static uint32_t const threads_per_block   = 256;

template <typename T>
__global__ void quad_pip_phase1_kernel(uint32_t const *pq_poly_id,
                                       uint32_t const *pq_quad_id,
                                       uint32_t const *block_offset,
                                       uint32_t const *block_length,
                                       cudf::column_device_view const quad_offsets,
                                       cudf::column_device_view const point_indices,
                                       cudf::column_device_view const point_x,
                                       cudf::column_device_view const point_y,
                                       cudf::column_device_view const poly_offsets,
                                       cudf::column_device_view const ring_offsets,
                                       cudf::column_device_view const poly_points_x,
                                       cudf::column_device_view const poly_points_y,
                                       uint32_t *num_hits)
{
  __shared__ uint32_t quad_idx, poly_idx, num_point, first_pos, qpos, num_adjusted;

  // assume # of points/threads no more than num_threads_per_warp * max_warps_per_block (32*32)
  __shared__ uint16_t data[max_warps_per_block];

  // assuming 1d
  if (threadIdx.x == 0) {
    quad_idx     = pq_quad_id[blockIdx.x];
    poly_idx     = pq_poly_id[blockIdx.x];
    qpos         = block_offset[blockIdx.x];
    num_point    = block_length[blockIdx.x];
    first_pos    = quad_offsets.element<uint32_t>(quad_idx) + qpos;
    num_adjusted = ((num_point - 1) / num_threads_per_warp + 1) * num_threads_per_warp;
  }

  __syncthreads();

  if (threadIdx.x < max_warps_per_block) { data[threadIdx.x] = 0; }

  __syncthreads();

  bool in_polygon   = false;
  uint32_t tid      = first_pos + threadIdx.x;
  uint32_t point_id = point_indices.element<uint32_t>(tid);

  if (threadIdx.x < num_point) {
    T x = point_x.element<T>(point_id);
    T y = point_y.element<T>(point_id);

    uint32_t poly_begin = poly_offsets.element<uint32_t>(poly_idx);
    uint32_t poly_end   = poly_idx < poly_offsets.size() - 1
                          ? poly_offsets.element<uint32_t>(poly_idx + 1)
                          : ring_offsets.size();

    for (uint32_t ring_idx = poly_begin; ring_idx < poly_end; ring_idx++)  // for each ring
    {
      auto ring_begin = ring_offsets.element<uint32_t>(ring_idx);
      auto ring_end   = ring_idx < ring_offsets.size() - 1
                        ? ring_offsets.element<uint32_t>(ring_idx + 1)
                        : poly_points_x.size();
      auto ring_len = ring_end - ring_begin;
      for (auto point_idx = 0; point_idx < ring_len; ++point_idx) {
        T x0                 = poly_points_x.element<T>(ring_begin + ((point_idx + 0) % ring_len));
        T y0                 = poly_points_y.element<T>(ring_begin + ((point_idx + 0) % ring_len));
        T x1                 = poly_points_x.element<T>(ring_begin + ((point_idx + 1) % ring_len));
        T y1                 = poly_points_y.element<T>(ring_begin + ((point_idx + 1) % ring_len));
        bool y_between_ay_by = y0 <= y && y < y1;  // is y in range [ay, by) when ay < by
        bool y_between_by_ay = y1 <= y && y < y0;  // is y in range [by, ay) when by < ay
        bool y_in_bounds     = y_between_ay_by || y_between_by_ay;  // is y in range [by, ay]
        T run                = x1 - x0;
        T rise               = y1 - y0;
        T rise_to_point      = y - y0;

        if (y_in_bounds && x < (run / rise) * rise_to_point + x0) { in_polygon = not in_polygon; }
      }
    }
  }

  __syncthreads();

  unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < num_adjusted);
  uint32_t vote = __ballot_sync(mask, in_polygon);

  if (threadIdx.x % num_threads_per_warp == 0) {
    data[threadIdx.x / num_threads_per_warp] = __popc(vote);
  }

  __syncthreads();

  if (threadIdx.x < max_warps_per_block) {
    uint32_t num = data[threadIdx.x];
    for (uint32_t offset = max_warps_per_block / 2; offset > 0; offset /= 2) {
      num += __shfl_xor_sync(0xFFFFFFFF, num, offset);
    }
    if (threadIdx.x == 0) { num_hits[blockIdx.x] = num; }
  }
}

template <typename T>
__global__ void quad_pip_phase2_kernel(const uint32_t *pq_poly_id,
                                       const uint32_t *pq_quad_id,
                                       uint32_t *block_offset,
                                       uint32_t *block_length,
                                       cudf::column_device_view const quad_offsets,
                                       cudf::column_device_view const point_indices,
                                       cudf::column_device_view const point_x,
                                       cudf::column_device_view const point_y,
                                       cudf::column_device_view const poly_offsets,
                                       cudf::column_device_view const ring_offsets,
                                       cudf::column_device_view const poly_points_x,
                                       cudf::column_device_view const poly_points_y,
                                       uint32_t *num_hits,
                                       cudf::mutable_column_device_view out_poly_idx,
                                       cudf::mutable_column_device_view out_point_idx)
{
  __shared__ uint32_t quad_idx, poly_idx, num_point, first_pos, mem_offset, qpos, num_adjusted;

  // assume # of points/threads no more than num_threads_per_warp * max_warps_per_block (32*32)
  __shared__ uint16_t temp[max_warps_per_block], sums[max_warps_per_block + 1];

  // assuming 1d
  if (threadIdx.x == 0) {
    quad_idx     = pq_quad_id[blockIdx.x];
    poly_idx     = pq_poly_id[blockIdx.x];
    qpos         = block_offset[blockIdx.x];
    num_point    = block_length[blockIdx.x];
    mem_offset   = num_hits[blockIdx.x];
    first_pos    = quad_offsets.element<uint32_t>(quad_idx) + qpos;
    sums[0]      = 0;
    num_adjusted = ((num_point - 1) / num_threads_per_warp + 1) * num_threads_per_warp;
  }

  __syncthreads();

  if (threadIdx.x < max_warps_per_block + 1) { temp[threadIdx.x] = 0; }

  __syncthreads();

  bool in_polygon   = false;
  uint32_t tid      = first_pos + threadIdx.x;
  uint32_t point_id = point_indices.element<uint32_t>(tid);

  if (threadIdx.x < num_point) {
    T x = point_x.element<T>(point_id);
    T y = point_y.element<T>(point_id);

    uint32_t poly_begin = poly_offsets.element<uint32_t>(poly_idx);
    uint32_t poly_end   = poly_idx < poly_offsets.size() - 1
                          ? poly_offsets.element<uint32_t>(poly_idx + 1)
                          : ring_offsets.size();

    for (uint32_t ring_idx = poly_begin; ring_idx < poly_end; ring_idx++)  // for each ring
    {
      auto ring_begin = ring_offsets.element<uint32_t>(ring_idx);
      auto ring_end   = ring_idx < ring_offsets.size() - 1
                        ? ring_offsets.element<uint32_t>(ring_idx + 1)
                        : poly_points_x.size();
      auto ring_len = ring_end - ring_begin;
      for (auto point_idx = 0; point_idx < ring_len; ++point_idx) {
        T x0                 = poly_points_x.element<T>(ring_begin + ((point_idx + 0) % ring_len));
        T y0                 = poly_points_y.element<T>(ring_begin + ((point_idx + 0) % ring_len));
        T x1                 = poly_points_x.element<T>(ring_begin + ((point_idx + 1) % ring_len));
        T y1                 = poly_points_y.element<T>(ring_begin + ((point_idx + 1) % ring_len));
        bool y_between_ay_by = y0 <= y && y < y1;  // is y in range [ay, by) when ay < by?
        bool y_between_by_ay = y1 <= y && y < y0;  // is y in range [by, ay) when by < ay?
        bool y_in_bounds     = y_between_ay_by || y_between_by_ay;  // is y in range [by, ay]?
        T run                = x1 - x0;
        T rise               = y1 - y0;
        T rise_to_point      = y - y0;

        if (y_in_bounds && x < (run / rise) * rise_to_point + x0) { in_polygon = not in_polygon; }
      }
    }
  }

  __syncthreads();

  unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < num_adjusted);
  uint32_t vote = __ballot_sync(mask, in_polygon);
  if (threadIdx.x % num_threads_per_warp == 0) {
    temp[threadIdx.x / num_threads_per_warp] = __popc(vote);
  }

  __syncthreads();

  // warp-level scan; only one warp is used
  if (threadIdx.x < num_threads_per_warp) {
    uint32_t num = temp[threadIdx.x];
    for (uint8_t i = 1; i <= num_threads_per_warp; i *= 2) {
      int n = __shfl_up_sync(0xFFFFFFF, num, i, num_threads_per_warp);
      if (threadIdx.x >= i) num += n;
    }
    sums[threadIdx.x + 1] = num;
  }

  __syncthreads();

  if ((threadIdx.x < num_point) && in_polygon) {
    uint16_t num         = sums[threadIdx.x / num_threads_per_warp];
    uint16_t warp_offset = __popc(vote >> (threadIdx.x % num_threads_per_warp)) - 1;
    uint32_t pos         = mem_offset + num + warp_offset;

    out_poly_idx.element<uint32_t>(pos)  = poly_idx;
    out_point_idx.element<uint32_t>(pos) = tid;
  }
}

template <typename T>
std::unique_ptr<cudf::table> compute_quadtree_point_in_polygon(
  cudf::column_view const &poly_indices,
  cudf::column_view const &quad_indices,
  cudf::column_view const &quad_lengths,
  cudf::column_view const &quad_offsets,
  cudf::column_view const &point_indices,
  cudf::column_view const &point_x,
  cudf::column_view const &point_y,
  cudf::column_view const &poly_offsets,
  cudf::column_view const &ring_offsets,
  cudf::column_view const &poly_points_x,
  cudf::column_view const &poly_points_y,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream)
{
  auto num_original_pairs = poly_indices.size();
  auto d_quad_lengths     = cudf::column_device_view::create(quad_lengths, stream);
  auto get_num_units      = [quad_lengths = *d_quad_lengths] __device__(uint32_t quad_index) {
    return ((quad_lengths.element<uint32_t>(quad_index) - 1) / threads_per_block) + 1;
  };

  // compute the total number of sub-pairs (units) using transform_reduce
  uint32_t num_unit_pairs = thrust::transform_reduce(rmm::exec_policy(stream)->on(stream),
                                                     quad_indices.begin<uint32_t>(),
                                                     quad_indices.end<uint32_t>(),
                                                     get_num_units,
                                                     0,
                                                     thrust::plus<uint32_t>());

  auto d_quad_offset = [&]() {
    // allocate memory for both numbers and their prefix-sums
    rmm::device_uvector<uint32_t> d_num_sums(num_original_pairs, stream);
    rmm::device_uvector<uint32_t> d_num_units(num_original_pairs, stream);

    // computes numbers of sub-pairs for each quadrant-polygon pair
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      quad_indices.begin<uint32_t>(),
                      quad_indices.end<uint32_t>(),
                      d_num_units.begin(),
                      get_num_units);

    thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                           d_num_units.begin(),
                           d_num_units.end(),
                           d_num_sums.begin());

    // allocate memory for sub-pairs with four components:
    // (polygon_idx, quadrant_idx, offset, length)

    rmm::device_uvector<uint32_t> d_quad_offset(num_unit_pairs, stream);

    thrust::fill(
      rmm::exec_policy(stream)->on(stream), d_quad_offset.begin(), d_quad_offset.end(), 0);

    // scatter 0..num_org_pair to d_quad_offset using d_num_sums as map
    thrust::scatter(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_original_pairs,
                    d_num_sums.begin(),
                    d_quad_offset.begin());

    // copy idx of orginal pairs to all sub-pairs
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           d_quad_offset.begin(),
                           d_quad_offset.end(),
                           d_quad_offset.begin(),
                           thrust::maximum<int>());

    // d_num_sums and d_num_units are no longer needed, so don't return them

    return std::move(d_quad_offset);
  }();

  rmm::device_uvector<uint32_t> d_pq_poly_idx(num_unit_pairs, stream);
  rmm::device_uvector<uint32_t> d_pq_quad_idx(num_unit_pairs, stream);
  rmm::device_uvector<uint32_t> d_quad_length(num_unit_pairs, stream);

  // gather polygon idx and quadrant idx from original pairs into sub-pairs
  thrust::gather(rmm::exec_policy(stream)->on(stream),
                 d_quad_offset.begin(),
                 d_quad_offset.end(),
                 poly_indices.begin<uint32_t>(),
                 d_pq_poly_idx.begin());

  thrust::gather(rmm::exec_policy(stream)->on(stream),
                 d_quad_offset.begin(),
                 d_quad_offset.end(),
                 quad_indices.begin<uint32_t>(),
                 d_pq_quad_idx.begin());

  // allocate memory to store numbers of points in polygons in all sub-pairs initialized to 0
  rmm::device_uvector<uint32_t> d_num_hits(num_unit_pairs, stream);
  thrust::fill(rmm::exec_policy(stream)->on(stream), d_num_hits.begin(), d_num_hits.end(), 0);

  // generate offsets of sub-paris within the orginal pairs
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                d_quad_offset.begin(),
                                d_quad_offset.end(),
                                thrust::constant_iterator<int>(1),
                                d_quad_offset.begin());

  // assemble components in input/output iterators; note d_quad_offset used in both input and output
  auto quad_block_id_iter = make_zip_iterator(d_pq_quad_idx.begin(), d_quad_offset.begin());
  auto offset_length_iter = make_zip_iterator(d_quad_offset.begin(), d_quad_length.begin());
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    quad_block_id_iter,
                    quad_block_id_iter + num_unit_pairs,
                    offset_length_iter,
                    [quad_lengths = *d_quad_lengths] __device__(auto const &v) {
                      uint32_t quad_id    = thrust::get<0>(v);
                      uint32_t block_id   = thrust::get<1>(v);
                      uint32_t num_points = quad_lengths.element<uint32_t>(quad_id);
                      uint32_t offset     = block_id * threads_per_block;
                      uint32_t length     = num_points < (block_id + 1) * threads_per_block
                                          ? num_points - block_id * threads_per_block
                                          : threads_per_block;
                      return thrust::make_tuple(offset, length);
                    });

  auto d_quad_offsets  = cudf::column_device_view::create(quad_offsets, stream);
  auto d_point_indices = cudf::column_device_view::create(point_indices, stream);
  auto d_point_x       = cudf::column_device_view::create(point_x, stream);
  auto d_point_y       = cudf::column_device_view::create(point_y, stream);
  auto d_poly_offsets  = cudf::column_device_view::create(poly_offsets, stream);
  auto d_ring_offsets  = cudf::column_device_view::create(ring_offsets, stream);
  auto d_poly_points_x = cudf::column_device_view::create(poly_points_x, stream);
  auto d_poly_points_y = cudf::column_device_view::create(poly_points_y, stream);

  quad_pip_phase1_kernel<T><<<num_unit_pairs, threads_per_block, 0, stream>>>(d_pq_poly_idx.data(),
                                                                              d_pq_quad_idx.data(),
                                                                              d_quad_offset.data(),
                                                                              d_quad_length.data(),
                                                                              *d_quad_offsets,
                                                                              *d_point_indices,
                                                                              *d_point_x,
                                                                              *d_point_y,
                                                                              *d_poly_offsets,
                                                                              *d_ring_offsets,
                                                                              *d_poly_points_x,
                                                                              *d_poly_points_y,
                                                                              d_num_hits.data());

  CUDA_TRY(cudaStreamSynchronize(stream));

  // remove poly-quad pair with zero hits
  auto valid_pq_pair_iter = make_zip_iterator(d_pq_poly_idx.begin(),
                                              d_pq_quad_idx.begin(),
                                              d_quad_offset.begin(),
                                              d_quad_length.begin(),
                                              d_num_hits.begin());

  uint32_t num_valid_pair = thrust::distance(
    valid_pq_pair_iter,
    thrust::remove_if(rmm::exec_policy(stream)->on(stream),
                      valid_pq_pair_iter,
                      valid_pq_pair_iter + num_unit_pairs,
                      valid_pq_pair_iter,
                      [] __device__(auto const &v) { return thrust::get<4>(v) == 0; }));

  d_pq_poly_idx.resize(num_valid_pair, stream);
  d_pq_quad_idx.resize(num_valid_pair, stream);
  d_quad_offset.resize(num_valid_pair, stream);
  d_quad_length.resize(num_valid_pair, stream);
  d_num_hits.resize(num_valid_pair, stream);

  uint32_t total_hits =
    thrust::reduce(rmm::exec_policy(stream)->on(stream), d_num_hits.begin(), d_num_hits.end());

  // prefix sum on numbers to generate offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream)->on(stream), d_num_hits.begin(), d_num_hits.end(), d_num_hits.begin());

  auto poly_index_col  = make_fixed_width_column<uint32_t>(total_hits, stream, mr);
  auto point_index_col = make_fixed_width_column<uint32_t>(total_hits, stream, mr);

  // write output directly to poly_index_col and point_index_col columns
  auto d_poly_index_col  = cudf::mutable_column_device_view::create(*poly_index_col, stream);
  auto d_point_index_col = cudf::mutable_column_device_view::create(*point_index_col, stream);

  quad_pip_phase2_kernel<T><<<num_valid_pair, threads_per_block, 0, stream>>>(d_pq_poly_idx.data(),
                                                                              d_pq_quad_idx.data(),
                                                                              d_quad_offset.data(),
                                                                              d_quad_length.data(),
                                                                              *d_quad_offsets,
                                                                              *d_point_indices,
                                                                              *d_point_x,
                                                                              *d_point_y,
                                                                              *d_poly_offsets,
                                                                              *d_ring_offsets,
                                                                              *d_poly_points_x,
                                                                              *d_poly_points_y,
                                                                              d_num_hits.data(),
                                                                              *d_poly_index_col,
                                                                              *d_point_index_col);

  CUDA_TRY(cudaStreamSynchronize(stream));

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(2);
  cols.push_back(std::move(poly_index_col));
  cols.push_back(std::move(point_index_col));
  return std::make_unique<cudf::table>(std::move(cols));
}

struct dispatch_quadtree_point_in_polygon {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  std::unique_ptr<cudf::table> operator()(cudf::table_view const &poly_quad_pairs,
                                          cudf::table_view const &quadtree,
                                          cudf::column_view const &point_indices,
                                          cudf::column_view const &point_x,
                                          cudf::column_view const &point_y,
                                          cudf::column_view const &poly_offsets,
                                          cudf::column_view const &ring_offsets,
                                          cudf::column_view const &poly_points_x,
                                          cudf::column_view const &poly_points_y,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
  {
    return compute_quadtree_point_in_polygon<T>(poly_quad_pairs.column(0),
                                                poly_quad_pairs.column(1),
                                                quadtree.column(3),
                                                quadtree.column(4),
                                                point_indices,
                                                point_x,
                                                point_y,
                                                poly_offsets,
                                                ring_offsets,
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

std::unique_ptr<cudf::table> quadtree_point_in_polygon(cudf::table_view const &poly_quad_pairs,
                                                       cudf::table_view const &quadtree,
                                                       cudf::column_view const &point_indices,
                                                       cudf::column_view const &point_x,
                                                       cudf::column_view const &point_y,
                                                       cudf::column_view const &poly_offsets,
                                                       cudf::column_view const &ring_offsets,
                                                       cudf::column_view const &poly_points_x,
                                                       cudf::column_view const &poly_points_y,
                                                       rmm::mr::device_memory_resource *mr,
                                                       cudaStream_t stream)
{
  return cudf::type_dispatcher(point_x.type(),
                               dispatch_quadtree_point_in_polygon{},
                               poly_quad_pairs,
                               quadtree,
                               point_indices,
                               point_x,
                               point_y,
                               poly_offsets,
                               ring_offsets,
                               poly_points_x,
                               poly_points_y,
                               mr,
                               stream);
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
  CUSPATIAL_EXPECTS(poly_offsets.size() > 0, "number of polygons must be greater than 0");
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
                                           mr,
                                           cudaStream_t{0});
}

}  // namespace cuspatial
