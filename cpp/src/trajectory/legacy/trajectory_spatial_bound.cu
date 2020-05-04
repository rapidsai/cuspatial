/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <type_traits>
#include <utilities/legacy/cuda_utils.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/legacy/trajectory.hpp>
#include <utility/trajectory_thrust.cuh>
#include <utility/utility.hpp>

#include <cudf/legacy/column.hpp>

namespace {

/**
 * @brief CUDA kernel for computing spatial bounding boxes of trajectories
 *
 */
template <typename T>
__global__ void sbbox_kernel(gdf_size_type num_traj,
                             const T* const __restrict__ x,
                             const T* const __restrict__ y,
                             const uint32_t* const __restrict__ len,
                             const uint32_t* const __restrict__ pos,
                             T* const __restrict__ bbox_x1,
                             T* const __restrict__ bbox_y1,
                             T* const __restrict__ bbox_x2,
                             T* const __restrict__ bbox_y2)
{
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= num_traj) return;
  int bp = (pid == 0) ? 0 : pos[pid - 1];
  int ep = pos[pid];

  bbox_x2[pid] = bbox_x1[pid] = x[bp];
  bbox_y2[pid] = bbox_y1[pid] = y[bp];

  for (int i = bp + 1; i < ep; i++) {
    if (bbox_x1[pid] > x[i]) bbox_x1[pid] = x[i];
    if (bbox_x2[pid] < x[i]) bbox_x2[pid] = x[i];
    if (bbox_y1[pid] > y[i]) bbox_y1[pid] = y[i];
    if (bbox_y2[pid] < y[i]) bbox_y2[pid] = y[i];
  }
}

struct sbbox_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_floating_point<T>::value;
  }

  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  void operator()(const gdf_column& x,
                  const gdf_column& y,
                  const gdf_column& length,
                  const gdf_column& offset,
                  gdf_column& bbox_x1,
                  gdf_column& bbox_y1,
                  gdf_column& bbox_x2,
                  gdf_column& bbox_y2)
  {
    T* temp{nullptr};
    RMM_TRY(RMM_ALLOC(&temp, length.size * sizeof(T), 0));

    gdf_column_view_augmented(&bbox_x1,
                              temp,
                              nullptr,
                              length.size,
                              x.dtype,
                              0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE},
                              "bbox_x1");

    RMM_TRY(RMM_ALLOC(&temp, length.size * sizeof(T), 0));
    gdf_column_view_augmented(&bbox_x2,
                              temp,
                              nullptr,
                              length.size,
                              x.dtype,
                              0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE},
                              "bbox_x2");

    RMM_TRY(RMM_ALLOC(&temp, length.size * sizeof(T), 0));
    gdf_column_view_augmented(&bbox_y1,
                              temp,
                              nullptr,
                              length.size,
                              x.dtype,
                              0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE},
                              "bbox_y1");

    RMM_TRY(RMM_ALLOC(&temp, length.size * sizeof(T), 0));
    gdf_column_view_augmented(&bbox_y2,
                              temp,
                              nullptr,
                              length.size,
                              x.dtype,
                              0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE},
                              "bbox_y2");

    gdf_size_type min_grid_size = 0, block_size = 0;
    CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, sbbox_kernel<T>));
    cudf::util::cuda::grid_config_1d grid{x.size, block_size, 1};
    sbbox_kernel<T><<<grid.num_blocks, block_size>>>(length.size,
                                                     static_cast<T*>(x.data),
                                                     static_cast<T*>(y.data),
                                                     static_cast<uint32_t*>(length.data),
                                                     static_cast<uint32_t*>(offset.data),
                                                     static_cast<T*>(bbox_x1.data),
                                                     static_cast<T*>(bbox_y1.data),
                                                     static_cast<T*>(bbox_x2.data),
                                                     static_cast<T*>(bbox_y2.data));
    CUDA_TRY(cudaDeviceSynchronize());
  }

  template <typename T, std::enable_if_t<!is_supported<T>()>* = nullptr>
  void operator()(const gdf_column& x,
                  const gdf_column& y,
                  const gdf_column& length,
                  const gdf_column& offset,
                  gdf_column& bbox_x1,
                  gdf_column& bbox_y1,
                  gdf_column& bbox_x2,
                  gdf_column& bbox_y2)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }
};

}  // namespace

namespace cuspatial {

/**
 * @brief computing spatial bounding boxes of trajectories
 *
 * see trajectory.hpp
 */
void trajectory_spatial_bounds(const gdf_column& x,
                               const gdf_column& y,
                               const gdf_column& length,
                               const gdf_column& offset,
                               gdf_column& bbox_x1,
                               gdf_column& bbox_y1,
                               gdf_column& bbox_x2,
                               gdf_column& bbox_y2)
{
  CUSPATIAL_EXPECTS(
    x.data != nullptr && y.data != nullptr && length.data != nullptr && offset.data != nullptr,
    "Null data pointer");
  CUSPATIAL_EXPECTS(x.size == y.size && length.size == offset.size, "Data size mismatch");

  // future versions might allow x/y/pos/len have null_count>0, which might be
  // useful for taking query results as inputs
  CUSPATIAL_EXPECTS(
    x.null_count == 0 && y.null_count == 0 && length.null_count == 0 && offset.null_count == 0,
    "Null data support not implemented");

  CUSPATIAL_EXPECTS(x.size >= offset.size, "one trajectory must have at least one point");

  cudf::type_dispatcher(
    x.dtype, sbbox_functor(), x, y, length, offset, bbox_x1, bbox_y1, bbox_x2, bbox_y2);

  // TODO: handle null_count if needed
}

}  // namespace cuspatial
