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
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <sys/time.h>
#include <time.h>

#include <utility/utility.hpp>
#include <utility/trajectory_thrust.cuh>
#include <cuspatial/trajectory.hpp>

namespace {
/**
 * @brief CUDA kernel for computing distances and speeds of trajectories
 *
 */
template <typename T>
__global__ void distspeed_kernel(gdf_size_type num_traj,
                                 const T* const __restrict__ x,
                                 const T* const __restrict__ y,
                                 const cuspatial::its_timestamp * const __restrict__ time,
                                 const uint32_t * const __restrict__ len,
                                 const uint32_t * const __restrict__ pos,
                                 T* const __restrict__ dis,
                                 T* const __restrict__ sp)
{
    int pid=blockIdx.x*blockDim.x+threadIdx.x;  
    if(pid>=num_traj) return;
    int bp=(pid==0)?0:pos[pid-1];
    int ep=pos[pid]-1;

    //assuming the same year --restriction to be removed 	 
    float td=(time[ep].yd-time[bp].yd)*86400;
    td+=(time[ep].hh-time[bp].hh)*3600;
    td+=(time[ep].mm-time[bp].mm)*60;
    td+=(time[ep].ss-time[bp].ss);
    td+=(time[ep].ms-time[bp].ms)/(float)1000; 	 

    if((len[pid]<2)||(td==0)||(time[ep].y!=time[bp].y)) 
    {
        dis[pid]=-1;
        sp[pid]=-1;
    }
    else
    {
        float ds=0;
        for(int i=0;i<len[pid]-1;i++)
        {
            float dt=(x[bp+i+1]-x[bp+i])*(x[bp+i+1]-x[bp+i]);
            dt+=(y[bp+i+1]-y[bp+i])*(y[bp+i+1]-y[bp+i]);
            ds+=sqrt(dt);
        }
        dis[pid]=ds*1000; //km to m
        sp[pid]=ds*1000/td; // m/s
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
                                  gdf_dtype_extra_info{TIME_UNIT_NONE}, "dist");

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
            static_cast<cuspatial::its_timestamp*>(timestamp.data),
            static_cast<uint32_t*>(length.data),
            static_cast<uint32_t*>(offset.data),
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
                 "Null data pointer");
    CUDF_EXPECTS(x.size == y.size && x.size == timestamp.size &&
                 length.size == offset.size, "Data size mismatch");

    //future versions might allow x/y/ts/pos/len have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(x.null_count == 0 && y.null_count == 0 &&
                 timestamp.null_count == 0 &&
                 length.null_count == 0 && offset.null_count == 0,
                 "Null data support not implemented");

    CUDF_EXPECTS(x.size >= offset.size ,
                 "one trajectory must have at least one point");

    std::pair<gdf_column,gdf_column> res_pair = 
        cudf::type_dispatcher(x.dtype, distspeed_functor(), x, y,
                              timestamp, length, offset);

    return res_pair;
}

}// namespace cuspatial

