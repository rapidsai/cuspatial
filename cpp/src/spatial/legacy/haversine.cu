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

#include <math.h>

#include <cudf/legacy/column.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <thrust/device_vector.h>
#include <cuspatial/error.hpp>
#include <cuspatial/legacy/haversine.hpp>
#include <type_traits>
#include <utilities/legacy/cuda_utils.hpp>
#include <utility/utility.hpp>

namespace {

template <typename T>
__global__ void haversine_distance_kernel(int pnt_size,
                                          const T* const __restrict__ x1,
                                          const T* const __restrict__ y1,
                                          const T* const __restrict__ x2,
                                          const T* const __restrict__ y2,
                                          T* const __restrict__ h_dist)
{
  // assuming 1D grid/block config
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pnt_size) return;
  T x_1       = M_PI / 180 * x1[idx];
  T y_1       = M_PI / 180 * y1[idx];
  T x_2       = M_PI / 180 * x2[idx];
  T y_2       = M_PI / 180 * y2[idx];
  T dlon      = x_2 - x_1;
  T dlat      = y_2 - y_1;
  T a         = sin(dlat / 2) * sin(dlat / 2) + cos(y_1) * cos(y_2) * sin(dlon / 2) * sin(dlon / 2);
  T c         = 2 * asin(sqrt(a));
  h_dist[idx] = c * 6371;
}

struct haversine_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_floating_point<T>::value;
  }

  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  gdf_column operator()(const gdf_column& x1,
                        const gdf_column& y1,
                        const gdf_column& x2,
                        const gdf_column& y2)
  {
    gdf_column h_dist;
    T* data{nullptr};

    cudaStream_t stream{0};
    RMM_TRY(RMM_ALLOC(&data, x1.size * sizeof(T), stream));
    gdf_column_view(&h_dist, data, nullptr, x1.size, x1.dtype);

    gdf_size_type min_grid_size = 0, block_size = 0;
    CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, haversine_distance_kernel<T>));
    cudf::util::cuda::grid_config_1d grid{x1.size, block_size, 1};

    haversine_distance_kernel<T><<<grid.num_blocks, block_size>>>(x1.size,
                                                                  static_cast<T*>(x1.data),
                                                                  static_cast<T*>(y1.data),
                                                                  static_cast<T*>(x2.data),
                                                                  static_cast<T*>(y2.data),
                                                                  static_cast<T*>(data));
    CUDA_TRY(cudaDeviceSynchronize());

    return h_dist;
  }

  template <typename T, std::enable_if_t<!is_supported<T>()>* = nullptr>
  gdf_column operator()(const gdf_column& x1,
                        const gdf_column& y1,
                        const gdf_column& x2,
                        const gdf_column& y2)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }
};

}  // namespace

/**
 *@brief Compute Haversine distances among pairs of logitude/latitude locations
 *see haversine.hpp
 */

namespace cuspatial {

/**
 * @brief Compute Haversine distances among pairs of logitude/latitude locations
 * see haversine.hpp
 */

gdf_column haversine_distance(const gdf_column& x1,
                              const gdf_column& y1,
                              const gdf_column& x2,
                              const gdf_column& y2)
{
  CUSPATIAL_EXPECTS(
    x1.data != nullptr && y1.data != nullptr && x2.data != nullptr && y2.data != nullptr,
    "point lon/lat cannot be empty");
  CUSPATIAL_EXPECTS(x1.dtype == x2.dtype && x2.dtype == y1.dtype && y1.dtype == y2.dtype,
                    "x1/x2/y1/y2 type mismatch");
  CUSPATIAL_EXPECTS(x1.size == x2.size && x2.size == y1.size && y1.size == y2.size,
                    "x1/x2/y1/y2 size mismatch");

  // future versions might allow pnt_(x/y) have null_count>0, which might be useful for taking query
  // results as inputs
  CUSPATIAL_EXPECTS(
    x1.null_count == 0 && y1.null_count == 0 && x2.null_count == 0 && y2.null_count == 0,
    "this version does not support x1/x2/y1/y2 contains nulls");

  gdf_column h_d = cudf::type_dispatcher(x1.dtype, haversine_functor(), x1, y1, x2, y2);

  return h_d;

}  // haversine_distance

}  // namespace cuspatial
