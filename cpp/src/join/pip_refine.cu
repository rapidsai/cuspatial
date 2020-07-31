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
#include "utility/join_thrust.cuh"

#include <cuspatial/error.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace cuspatial {
namespace detail {
namespace {

template <typename T>
__global__ void quad_pip_phase1_kernel(const uint32_t *pq_poly_id,
                                       const uint32_t *pq_quad_id,
                                       const uint32_t *block_offset,
                                       const uint32_t *block_length,
                                       const uint32_t *qt_fpos,
                                       const T *point_x,
                                       const T *point_y,
                                       const uint32_t *poly_fpos,
                                       const uint32_t *poly_rpos,
                                       const T *poly_x,
                                       const T *poly_y,
                                       uint32_t *num_hits)
{
  __shared__ uint32_t qid, pid, num_point, first_pos, qpos, num_adjusted;

  // assume #of points/threads no more than num_threads_per_warp*max_warps_per_block (32*32)
  __shared__ uint16_t data[max_warps_per_block];

  // assuming 1d
  if (threadIdx.x == 0) {
    qid          = pq_quad_id[blockIdx.x];
    pid          = pq_poly_id[blockIdx.x];
    qpos         = block_offset[blockIdx.x];
    num_point    = block_length[blockIdx.x];
    first_pos    = qt_fpos[qid] + qpos;
    num_adjusted = ((num_point - 1) / num_threads_per_warp + 1) * num_threads_per_warp;
    // printf("block=%d qid=%d pid=%d num_point=%d first_pos=%d\n",
    // blockIdx.x,qid,pid,num_point,first_pos);
  }
  __syncthreads();

  if (threadIdx.x < max_warps_per_block) data[threadIdx.x] = 0;
  __syncthreads();

  uint32_t tid    = first_pos + threadIdx.x;
  bool in_polygon = false;
  if (threadIdx.x < num_point) {
    T x = point_x[tid];
    T y = point_y[tid];

    uint32_t r_f = (0 == pid) ? 0 : poly_fpos[pid - 1];
    uint32_t r_t = poly_fpos[pid];
    for (uint32_t k = r_f; k < r_t; k++)  // for each ring
    {
      uint32_t m = (k == 0) ? 0 : poly_rpos[k - 1];
      for (; m < poly_rpos[k] - 1; m++)  // for each line segment
      {
        T x0, x1, y0, y1;
        x0 = poly_x[m];
        y0 = poly_y[m];
        x1 = poly_x[m + 1];
        y1 = poly_y[m + 1];
        // printf("block=%2d thread=%2d tid=%2d r_f=%2d r_t=%2d x=%10.5f y=%10.5f x0=%10.5f
        // y0=%10.5f x1=%10.5f y1=%10.5f\n",
        //    blockIdx.x,threadIdx.x,tid,r_f,r_t,x,y,x0,y0,x1,y1);

        if ((((y0 <= y) && (y < y1)) || ((y1 <= y) && (y < y0))) &&
            (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0))
          in_polygon = !in_polygon;
      }  // m
    }    // k
  }
  __syncthreads();

  unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < num_adjusted);
  uint32_t vote = __ballot_sync(mask, in_polygon);
  // printf("p1: block=%d thread=%d tid=%d in_polygon=%d mask=%08x
  // vote=%08x\n",blockIdx.x,threadIdx.x,tid,in_polygon,mask,vote);

  if (threadIdx.x % num_threads_per_warp == 0)
    data[threadIdx.x / num_threads_per_warp] = __popc(vote);
  __syncthreads();

  if (threadIdx.x < max_warps_per_block) {
    uint32_t num = data[threadIdx.x];
    for (uint32_t offset = max_warps_per_block / 2; offset > 0; offset /= 2)
      num += __shfl_xor_sync(0xFFFFFFFF, num, offset);
    if (threadIdx.x == 0) num_hits[blockIdx.x] = num;
  }
  __syncthreads();
}

template <typename T>
__global__ void quad_pip_phase2_kernel(const uint32_t *pq_poly_id,
                                       const uint32_t *pq_quad_id,
                                       uint32_t *block_offset,
                                       uint32_t *block_length,
                                       const uint32_t *qt_fpos,
                                       const T *point_x,
                                       const T *point_y,
                                       const uint32_t *poly_fpos,
                                       const uint32_t *poly_rpos,
                                       const T *poly_x,
                                       const T *poly_y,
                                       uint32_t *d_num_hits,
                                       uint32_t *d_res_poly_id,
                                       uint32_t *d_res_point_id)
{
  __shared__ uint32_t qid, pid, num_point, first_pos, mem_offset, qpos, num_adjusted;

  // assume #of points/threads no more than num_threads_per_warp*max_warps_per_block (32*32)
  __shared__ uint16_t temp[max_warps_per_block], sums[max_warps_per_block + 1];

  // assuming 1d
  if (threadIdx.x == 0) {
    qid          = pq_quad_id[blockIdx.x];
    pid          = pq_poly_id[blockIdx.x];
    qpos         = block_offset[blockIdx.x];
    num_point    = block_length[blockIdx.x];
    mem_offset   = d_num_hits[blockIdx.x];
    first_pos    = qt_fpos[qid] + qpos;
    sums[0]      = 0;
    num_adjusted = ((num_point - 1) / num_threads_per_warp + 1) * num_threads_per_warp;
    // printf("block=%d qid=%d pid=%d num_point=%d first_pos=%d mem_offset=%d\n",
    //    blockIdx.x,qid,pid,num_point,first_pos,mem_offset);
  }
  __syncthreads();

  if (threadIdx.x < max_warps_per_block + 1) temp[threadIdx.x] = 0;
  __syncthreads();

  uint32_t tid    = first_pos + threadIdx.x;
  bool in_polygon = false;
  if (threadIdx.x < num_point) {
    T x = point_x[tid];
    T y = point_y[tid];

    uint32_t r_f = (0 == pid) ? 0 : poly_fpos[pid - 1];
    uint32_t r_t = poly_fpos[pid];
    for (uint32_t k = r_f; k < r_t; k++)  // for each ring
    {
      uint32_t m = (k == 0) ? 0 : poly_rpos[k - 1];
      for (; m < poly_rpos[k] - 1; m++)  // for each line segment
      {
        T x0, x1, y0, y1;
        x0 = poly_x[m];
        y0 = poly_y[m];
        x1 = poly_x[m + 1];
        y1 = poly_y[m + 1];

        if ((((y0 <= y) && (y < y1)) || ((y1 <= y) && (y < y0))) &&
            (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0))
          in_polygon = !in_polygon;
      }  // m
    }    // k
  }
  __syncthreads();

  unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < num_adjusted);
  uint32_t vote = __ballot_sync(mask, in_polygon);
  if (threadIdx.x % num_threads_per_warp == 0)
    temp[threadIdx.x / num_threads_per_warp] = __popc(vote);
  __syncthreads();

  // warp-level scan; only one warp is used
  if (threadIdx.x < num_threads_per_warp) {
    uint32_t num = temp[threadIdx.x];
    for (uint8_t i = 1; i <= num_threads_per_warp; i *= 2) {
      int n = __shfl_up_sync(0xFFFFFFF, num, i, num_threads_per_warp);
      if (threadIdx.x >= i) num += n;
    }
    sums[threadIdx.x + 1] = num;
    __syncthreads();
  }
  // important!!!!!!!!!!!
  __syncthreads();

  if ((threadIdx.x < num_point) && (in_polygon)) {
    uint16_t num         = sums[threadIdx.x / num_threads_per_warp];
    uint16_t warp_offset = __popc(vote >> (threadIdx.x % num_threads_per_warp)) - 1;
    uint32_t pos         = mem_offset + num + warp_offset;

    // printf("block=%d thread=%d qid=%d pid=%d tid=%d mem_offset=%d num=%d warp_offset=%d
    // pos=%d\n",
    //    blockIdx.x,threadIdx.x,qid,pid,tid,mem_offset,num,warp_offset,pos);

    d_res_poly_id[pos]  = pid;
    d_res_point_id[pos] = tid;
  }
  __syncthreads();
}

template <typename T>
std::unique_ptr<cudf::table> dowork(uint32_t num_org_pair,
                                    cudf::column_view const &poly_idx,
                                    cudf::column_view const &quad_idx,
                                    uint32_t num_node,
                                    cudf::column_view const &qt_length,
                                    cudf::column_view const &qt_fpos,
                                    const uint32_t num_point,
                                    cudf::column_view const &point_x,
                                    cudf::column_view const &point_y,
                                    const uint32_t num_poly,
                                    cudf::column_view const &poly_fpos,
                                    cudf::column_view const &poly_rpos,
                                    cudf::column_view const &poly_x,
                                    cudf::column_view const &poly_y,
                                    rmm::mr::device_memory_resource *mr,
                                    cudaStream_t stream)

{
  // compute the total number of sub-pairs (units) using transform_reduce
  uint32_t num_pq_pair =
    thrust::transform_reduce(rmm::exec_policy(stream)->on(stream),
                             quad_idx.begin<uint32_t>(),
                             quad_idx.end<uint32_t>(),
                             get_num_units(qt_length.data<uint32_t>(), threads_per_block),
                             0,
                             thrust::plus<uint32_t>());

  // std::cout << "num_pq_pair=" << num_pq_pair << std::endl;

  auto poly_quad_offsets_lengths = [&]() {
    // allocate memory for both numbers and their prefix-sums
    rmm::device_uvector<uint32_t> d_num_sums(num_org_pair, stream);
    rmm::device_uvector<uint32_t> d_num_units(num_org_pair, stream);

    // computes numbers of sub-pairs for each quadrant-polygon pairs
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      quad_idx.begin<uint32_t>(),
                      quad_idx.end<uint32_t>(),
                      d_num_units.begin(),
                      get_num_units(qt_length.data<uint32_t>(), threads_per_block));

    // if (1) {
    //   print<uint32_t>(poly_idx, std::cout << "pre: d_org_poly_idx ", ", ", stream);
    //   print<uint32_t>(quad_idx, std::cout << "pre: d_org_quad_idx ", ", ", stream);
    //   print(d_num_units, std::cout << "pre: d_num_units ", ", ", stream);
    // }

    thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                           d_num_units.begin(),
                           d_num_units.end(),
                           d_num_sums.begin());

    // if (1) {
    //   print(d_num_sums, std::cout << "pre: d_num_sums ", ", ", stream);
    // }

    // allocate memory for sub-pairs with four components:
    // (polygon_idx, quadrant_idx, offset, length)

    rmm::device_uvector<uint32_t> d_pq_poly_idx(num_pq_pair, stream);
    rmm::device_uvector<uint32_t> d_pq_quad_idx(num_pq_pair, stream);
    rmm::device_uvector<uint32_t> d_quad_offset(num_pq_pair, stream);
    rmm::device_uvector<uint32_t> d_quad_length(num_pq_pair, stream);

    thrust::fill(
      rmm::exec_policy(stream)->on(stream), d_quad_offset.begin(), d_quad_offset.end(), 0);

    // scatter 0..num_org_pair to d_quad_offset using d_num_sums as map
    thrust::scatter(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_org_pair,
                    d_num_sums.begin(),
                    d_quad_offset.begin());

    // if (1) {
    //   print(d_quad_offset, std::cout << "pre: d_quad_offset (after scatter) ", ", ",
    //   stream);
    // }

    // copy idx of orginal pairs to all sub-pairs
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           d_quad_offset.begin(),
                           d_quad_offset.end(),
                           d_quad_offset.begin(),
                           thrust::maximum<int>());

    // d_num_sums and d_num_units are no longer needed, so don't return them

    return std::make_tuple(std::move(d_pq_poly_idx),
                           std::move(d_pq_quad_idx),
                           std::move(d_quad_offset),
                           std::move(d_quad_length));
  }();

  auto &d_pq_poly_idx = std::get<0>(poly_quad_offsets_lengths);
  auto &d_pq_quad_idx = std::get<1>(poly_quad_offsets_lengths);
  auto &d_quad_offset = std::get<2>(poly_quad_offsets_lengths);
  auto &d_quad_length = std::get<3>(poly_quad_offsets_lengths);

  // if (1) {
  //   print(d_quad_offset, std::cout << "pre: d_quad_offset (after scan) ", ", ", stream);
  // }

  // gather polygon idx and quadrant idx from original pairs into sub-pairs
  thrust::gather(rmm::exec_policy(stream)->on(stream),
                 d_quad_offset.begin(),
                 d_quad_offset.end(),
                 poly_idx.begin<uint32_t>(),
                 d_pq_poly_idx.begin());

  thrust::gather(rmm::exec_policy(stream)->on(stream),
                 d_quad_offset.begin(),
                 d_quad_offset.end(),
                 quad_idx.begin<uint32_t>(),
                 d_pq_quad_idx.begin());

  // if (1) {
  //   print(d_pq_poly_idx, std::cout << "pre: d_pq_poly_idx (after gather) ", ", ", stream);
  //   print(d_pq_quad_idx, std::cout << "pre: d_pq_quad_idx (after gather) ", ", ", stream);
  // }

  // allocate memory to store numbers of points in polygons in all sub-pairs initialized to 0
  rmm::device_uvector<uint32_t> d_num_hits(num_pq_pair, stream);
  thrust::fill(rmm::exec_policy(stream)->on(stream), d_num_hits.begin(), d_num_hits.end(), 0);

  // generate offsets of sub-paris within the orginal pairs
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                d_quad_offset.begin(),
                                d_quad_offset.end(),
                                thrust::constant_iterator<int>(1),
                                d_quad_offset.begin());

  // assemble components in input/output iterators; note d_quad_offset used in both input and output
  auto qid_bid_iter       = make_zip_iterator(d_pq_quad_idx.begin(), d_quad_offset.begin());
  auto offset_length_iter = make_zip_iterator(d_quad_offset.begin(), d_quad_length.begin());
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    qid_bid_iter,
                    qid_bid_iter + num_pq_pair,
                    offset_length_iter,
                    gen_offset_length(threads_per_block, qt_length.data<uint32_t>()));

  // if (1) {
  //   print(d_pq_poly_idx, std::cout << "pre: d_pq_poly_idx (complete result) ", ", ", stream);
  //   print(d_pq_quad_idx, std::cout << "pre: d_pq_quad_idx (complete result) ", ", ", stream);
  //   print(d_quad_offset, std::cout << "pre: d_quad_offset (complete result) ", ", ", stream);
  //   print(d_quad_length, std::cout << "pre: d_quad_length (complete result) ", ", ", stream);
  // }

  // std::cout << "running quad_pip_phase1_kernel" << std::endl;

  quad_pip_phase1_kernel<T>
    <<<num_pq_pair, threads_per_block, 0, stream>>>(d_pq_poly_idx.data(),
                                                    d_pq_quad_idx.data(),
                                                    d_quad_offset.data(),
                                                    d_quad_length.data(),
                                                    qt_fpos.data<uint32_t>(),
                                                    point_x.data<T>(),
                                                    point_y.data<T>(),
                                                    poly_fpos.data<uint32_t>(),
                                                    poly_rpos.data<uint32_t>(),
                                                    poly_x.data<T>(),
                                                    poly_y.data<T>(),
                                                    d_num_hits.data());

  CUDA_TRY(cudaStreamSynchronize(stream));

  // if (1) {
  //   print(d_num_hits, std::cout << "phase1: d_num_hits (before reduce) ", ", ", stream);
  // }

  // remove poly-quad pair with zero hits
  auto valid_pq_pair_iter = make_zip_iterator(d_pq_poly_idx.begin(),
                                              d_pq_quad_idx.begin(),
                                              d_quad_offset.begin(),
                                              d_quad_length.begin(),
                                              d_num_hits.begin());
  uint32_t num_valid_pair = thrust::distance(valid_pq_pair_iter,
                                             thrust::remove_if(rmm::exec_policy(stream)->on(stream),
                                                               valid_pq_pair_iter,
                                                               valid_pq_pair_iter + num_pq_pair,
                                                               valid_pq_pair_iter,
                                                               pq_remove_zero()));

  // std::cout << "num_valid_pair=" << num_valid_pair << std::endl;

  d_pq_poly_idx.resize(num_valid_pair, stream);
  d_pq_quad_idx.resize(num_valid_pair, stream);
  d_quad_offset.resize(num_valid_pair, stream);
  d_quad_length.resize(num_valid_pair, stream);
  d_num_hits.resize(num_valid_pair, stream);

  d_pq_poly_idx.shrink_to_fit(stream);
  d_pq_quad_idx.shrink_to_fit(stream);
  d_quad_offset.shrink_to_fit(stream);
  d_quad_length.shrink_to_fit(stream);
  d_num_hits.shrink_to_fit(stream);

  // if (1) {
  //   print(d_num_hits, std::cout << "phase1: d_num_hits (after removal) ", ", ", stream);
  // }

  uint32_t total_hits =
    thrust::reduce(rmm::exec_policy(stream)->on(stream), d_num_hits.begin(), d_num_hits.end());

  // std::cout << "total_hits=" << total_hits << std::endl;

  // prefix sum on numbers to generate offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream)->on(stream), d_num_hits.begin(), d_num_hits.end(), d_num_hits.begin());

  // if (1) {
  //   print(d_num_hits, std::cout << "phase1: d_num_hits (after reduce) ", ", ", stream);
  // }

  // use arrays in poly_idx and point_idx columns as kernel arguments to directly write output to
  // columns
  auto poly_idx_col  = make_fixed_width_column<uint32_t>(total_hits, stream, mr);
  auto point_idx_col = make_fixed_width_column<uint32_t>(total_hits, stream, mr);

  // std::cout << "running quad_pip_phase2_kernel" << std::endl;

  quad_pip_phase2_kernel<T><<<num_valid_pair, threads_per_block, 0, stream>>>(
    d_pq_poly_idx.data(),
    d_pq_quad_idx.data(),
    d_quad_offset.data(),
    d_quad_length.data(),
    qt_fpos.data<uint32_t>(),
    point_x.data<T>(),
    point_y.data<T>(),
    poly_fpos.data<uint32_t>(),
    poly_rpos.data<uint32_t>(),
    poly_x.data<T>(),
    poly_y.data<T>(),
    d_num_hits.data(),
    poly_idx_col->mutable_view().data<uint32_t>(),
    point_idx_col->mutable_view().data<uint32_t>());

  CUDA_TRY(cudaStreamSynchronize(stream));

  // if (1) {
  //   print<uint32_t>(*poly_idx_col, std::cout << "phase2: d_res_poly_id ", ", ", stream);
  //   print<uint32_t>(*point_idx_col, std::cout << "phase2: d_res_point_idx ", ", ", stream);
  // }

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(5);
  cols.push_back(std::move(poly_idx_col));
  cols.push_back(std::move(point_idx_col));
  return std::make_unique<cudf::table>(std::move(cols));
}

struct pip_refine_processor {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  std::unique_ptr<cudf::table> operator()(cudf::table_view const &pq_pair,
                                          cudf::table_view const &quadtree,
                                          cudf::table_view const &point,
                                          cudf::column_view const &poly_fpos,
                                          cudf::column_view const &poly_rpos,
                                          cudf::column_view const &poly_x,
                                          cudf::column_view const &poly_y,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
  {
    uint32_t num_pair  = pq_pair.num_rows();
    uint32_t num_node  = quadtree.num_rows();
    uint32_t num_poly  = poly_fpos.size();
    uint32_t num_point = point.num_rows();

    return dowork<T>(num_pair,
                     pq_pair.column(0),
                     pq_pair.column(1),
                     num_node,
                     quadtree.column(3),
                     quadtree.column(4),
                     num_point,
                     point.column(0),
                     point.column(1),
                     num_poly,
                     poly_fpos,
                     poly_rpos,
                     poly_x,
                     poly_y,
                     mr,
                     stream);
  }

  template <typename T, std::enable_if_t<!std::is_floating_point<T>::value> * = nullptr>
  std::unique_ptr<cudf::table> operator()(cudf::table_view const &pq_pair,
                                          cudf::table_view const &quadtree,
                                          cudf::table_view const &point,
                                          cudf::column_view const &poly_fpos,
                                          cudf::column_view const &poly_rpos,
                                          cudf::column_view const &poly_x,
                                          cudf::column_view const &poly_y,
                                          rmm::mr::device_memory_resource *mr,
                                          cudaStream_t stream)
  {
    CUDF_FAIL("Non-floating point operation is not supported");
  }
};

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::table> pip_refine(cudf::table_view const &pq_pair,
                                        cudf::table_view const &quadtree,
                                        cudf::table_view const &point,
                                        cudf::column_view const &poly_fpos,
                                        cudf::column_view const &poly_rpos,
                                        cudf::column_view const &poly_x,
                                        cudf::column_view const &poly_y,
                                        rmm::mr::device_memory_resource *mr)
{
  CUSPATIAL_EXPECTS(pq_pair.num_columns() == 2, "a quadrant-polygon table must have 2 columns");
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "a quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(point.num_columns() == 2, "a point table must have 5 columns");
  CUSPATIAL_EXPECTS(poly_fpos.size() > 0, "number of polygons must be greater than 0");
  CUSPATIAL_EXPECTS(poly_rpos.size() >= poly_fpos.size(),
                    "number of rings must be no less than number of polygons");
  CUSPATIAL_EXPECTS(poly_x.size() == poly_y.size(),
                    "numbers of vertices must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(poly_x.size() >= 4 * poly_rpos.size(),
                    "all rings must have at least 4 vertices");

  cudf::data_type point_dtype = point.column(0).type();
  cudf::data_type poly_dtype  = poly_x.type();
  CUSPATIAL_EXPECTS(point_dtype == poly_dtype, "point and polygon must have the same data type");

  return cudf::type_dispatcher(point_dtype,
                               detail::pip_refine_processor{},
                               pq_pair,
                               quadtree,
                               point,
                               poly_fpos,
                               poly_rpos,
                               poly_x,
                               poly_y,
                               mr,
                               cudaStream_t{0});
}

}  // namespace cuspatial
