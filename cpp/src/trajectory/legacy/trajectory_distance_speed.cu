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
#include <utilities/legacy/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>

#include <cuspatial/legacy/trajectory.hpp>

#include <cudf/legacy/column.hpp>

namespace {

/*
 * CUDA kernel for computing distances and speeds of trajectories
 */
template <typename T>
__global__ void distspeed_kernel(gdf_size_type num_traj,
                                 const T* const __restrict__ x,
                                 const T* const __restrict__ y,
                                 const cudf::timestamp * const __restrict__ time,
                                 const int32_t * const __restrict__ length,
                                 const int32_t * const __restrict__ pos,
                                 T* const __restrict__ dis,
                                 T* const __restrict__ sp)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_traj) return;
    int bp= (pid == 0) ? 0 : pos[pid - 1];
    int ep=pos[pid]-1;

    cudf::timestamp b = time[bp];
    cudf::timestamp e = time[ep];
    cudf::timestamp td = e - b;

    if(length[pid] < 2)
    {
        dis[pid] = -2;
        sp[pid] = -2;
    }
    else if(unwrap(td)==0)
    {
        dis[pid] = -3;
        sp[pid] = -3;
    }
    else
    {
        T ds=0;
        for(int i = 0; i < length[pid]-1; i++)
        {
            T dx = (x[bp + i + 1] - x[bp + i]);
            T dy = (y[bp + i + 1] - y[bp + i]);
            ds+=sqrt(dx*dx + dy*dy);
        }
        dis[pid] = ds * 1000; //km to m
        sp[pid] = ds * 1000000 / unwrap(td); // m/s
    }
}

struct distspeed_functor
{
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<T>::value;
    }

    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    std::pair<gdf_column,gdf_column> operator()(const gdf_column& x,
                                                const gdf_column& y,
                                                const gdf_column& timestamp,
                                                const gdf_column& length,
                                                const gdf_column& offset)
    {
        gdf_column dist{};
        T* temp{nullptr};
        RMM_TRY( RMM_ALLOC(&temp, length.size * sizeof(T), 0) );
        gdf_column_view_augmented(&dist, temp, nullptr, length.size, x.dtype, 0,
                                  gdf_dtype_extra_info{TIME_UNIT_NONE}, "distance");

        gdf_column speed{};
        RMM_TRY( RMM_ALLOC(&temp, length.size * sizeof(T), 0) );
        gdf_column_view_augmented(&speed, temp, nullptr, length.size, x.dtype, 0,
                                  gdf_dtype_extra_info{TIME_UNIT_NONE}, "speed");

        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
                                                     &block_size,
                                                     distspeed_kernel<T>) );
        cudf::util::cuda::grid_config_1d grid{x.size, block_size, 1};
        distspeed_kernel<T><<<grid.num_blocks, block_size>>>(length.size,
            static_cast<T*>(x.data), static_cast<T*>(y.data),
            static_cast<cudf::timestamp*>(timestamp.data),
            static_cast<int32_t*>(length.data),
            static_cast<int32_t*>(offset.data),
            static_cast<T*>(dist.data), static_cast<T*>(speed.data) );
        CUDA_TRY( cudaDeviceSynchronize() );

        return std::make_pair(dist,speed);
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr>
    std::pair<gdf_column,gdf_column> operator()(const gdf_column& x,
                                                const gdf_column& y,
                                                const gdf_column& timestamp,
                                                const gdf_column& length,
                                                const gdf_column& offset)
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};

} // namespace anonymous


namespace cuspatial {

/*
 * Compute distance(length) and speed of trajectories
 *
 * see trajectory.hpp
 */
std::pair<gdf_column,gdf_column>
trajectory_distance_and_speed(const gdf_column& x, const gdf_column& y,
                              const gdf_column& timestamp,
                              const gdf_column& length,
                              const gdf_column& offset)
{

    CUDF_EXPECTS(x.data != nullptr && y.data != nullptr &&
                 timestamp.data != nullptr && length.data != nullptr &&
                 offset.data != nullptr,
                 "Null input data");
    CUDF_EXPECTS(x.size == y.size && x.size == timestamp.size &&
                 length.size == offset.size, "Data size mismatch");
    CUDF_EXPECTS(timestamp.dtype == GDF_TIMESTAMP,
                 "Invalid timestamp datatype");
    CUDF_EXPECTS(length.dtype == GDF_INT32,
                 "Invalid trajectory length datatype");
    CUDF_EXPECTS(offset.dtype == GDF_INT32,
                 "Invalid trajectory offset datatype");
    CUDF_EXPECTS(x.null_count == 0 && y.null_count == 0 &&
                 timestamp.null_count == 0 &&
                 length.null_count == 0 && offset.null_count == 0,
                 "NULL support unimplemented");
    CUDF_EXPECTS(x.size >= offset.size ,
                 "Insufficient trajectory data");

    std::pair<gdf_column,gdf_column> res_pair =
        cudf::type_dispatcher(x.dtype, distspeed_functor(), x, y,
                              timestamp, length, offset);

    return res_pair;
}

}// namespace cuspatial
