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

#include <thrust/device_vector.h>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <type_traits>
#include <utilities/legacy/cuda_utils.hpp>
#include <utility>

#include <cuspatial/error.hpp>
#include <cuspatial/legacy/coordinate_transform.hpp>
#include <utility/utility.hpp>

#include <cudf/legacy/column.hpp>

namespace {

/**
 * @brief CUDA kernel for approximately transforming lon/lat to x/y (in km) relative to a camera
 *origin
 *
 *Note: points in the third quadrant relative camera will be transformed into points in the first
 *quadrant - use with caution
 */

template <typename T>
__global__ void coord_trans_kernel(gdf_size_type loc_size,
                                   double cam_lon,
                                   double cam_lat,
                                   const T* const __restrict__ in_lon,
                                   const T* const __restrict__ in_lat,
                                   T* const __restrict__ out_x,
                                   T* const __restrict__ out_y)
{
  // assuming 1D grid/block config
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= loc_size) return;
  out_x[idx] =
    ((cam_lon - in_lon[idx]) * 40000.0 * cos((cam_lat + in_lat[idx]) * M_PI / 360) / 360);
  out_y[idx] = (cam_lat - in_lat[idx]) * 40000.0 / 360;
}

struct ll2coord_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_floating_point<T>::value;
  }

  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  std::pair<gdf_column, gdf_column> operator()(const gdf_scalar& cam_lon,
                                               const gdf_scalar& cam_lat,
                                               const gdf_column& in_lon,
                                               const gdf_column& in_lat)

  {
    gdf_column out_x, out_y;
    memset(&out_x, 0, sizeof(gdf_column));
    memset(&out_y, 0, sizeof(gdf_column));

    cudaStream_t stream{0};
    T* temp_x{nullptr};
    T* temp_y{nullptr};
    RMM_TRY(RMM_ALLOC(&temp_x, in_lon.size * sizeof(T), stream));
    RMM_TRY(RMM_ALLOC(&temp_y, in_lat.size * sizeof(T), stream));

    gdf_column_view_augmented(&out_x,
                              temp_x,
                              nullptr,
                              in_lon.size,
                              in_lon.dtype,
                              0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE},
                              "x");
    gdf_column_view_augmented(&out_y,
                              temp_y,
                              nullptr,
                              in_lat.size,
                              in_lat.dtype,
                              0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE},
                              "y");

    gdf_size_type min_grid_size = 0, block_size = 0;
    CUDA_TRY(
      cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, coord_trans_kernel<T>));
    cudf::util::cuda::grid_config_1d grid{in_lon.size, block_size, 1};

    coord_trans_kernel<T><<<grid.num_blocks, block_size>>>(in_lon.size,
                                                           *((double*)(&(cam_lon.data))),
                                                           *((double*)(&(cam_lat.data))),
                                                           static_cast<T*>(in_lon.data),
                                                           static_cast<T*>(in_lat.data),
                                                           temp_x,
                                                           temp_y);
    CUDA_TRY(cudaDeviceSynchronize());

    return std::make_pair(out_x, out_y);
  }

  template <typename T, std::enable_if_t<!is_supported<T>()>* = nullptr>
  std::pair<gdf_column, gdf_column> operator()(const gdf_scalar& cam_lon,
                                               const gdf_scalar& cam_lat,
                                               const gdf_column& in_lon,
                                               const gdf_column& in_lat)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }
};

}  // namespace

namespace cuspatial {

/**
 * @brief transforming in_lon/in_lat (lon/lat defined in coord_2d) to out_x/out_y relative to a
 * camera origiin see coordinate_transform.hpp
 */

std::pair<gdf_column, gdf_column> lonlat_to_coord(const gdf_scalar& cam_lon,
                                                  const gdf_scalar& cam_lat,
                                                  const gdf_column& in_lon,
                                                  const gdf_column& in_lat)

{
  double cx = *((double*)(&(cam_lon.data)));
  double cy = *((double*)(&(cam_lat.data)));
  CUSPATIAL_EXPECTS(cx >= -180 && cx <= 180 && cy >= -90 && cy <= 90,
                    "camera origin must have valid lat/lon values [-180,-90,180,90]");
  CUSPATIAL_EXPECTS(in_lon.data != nullptr && in_lat.data != nullptr,
                    "input point cannot be empty");
  CUSPATIAL_EXPECTS(in_lon.size == in_lat.size, "input x and y arrays must have the same length");

  // future versions might allow in_(x/y) have null_count>0, which might be useful for taking query
  // results as inputs
  CUSPATIAL_EXPECTS(in_lon.null_count == 0 && in_lat.null_count == 0,
                    "this version does not support point in_lon/in_lat contains nulls");

  auto res =
    cudf::type_dispatcher(in_lon.dtype, ll2coord_functor(), cam_lon, cam_lat, in_lon, in_lat);

  return res;

}  // lonlat_to_coord

}  // namespace cuspatial
