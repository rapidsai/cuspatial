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

#include <sys/time.h>
#include <time.h>

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <utility/utility.hpp>
#include <cuspatial/haversine.hpp>

using namespace std; 
using namespace cudf;
using namespace cuspatial;

#define pi 3.1415926535 

 template <typename T>
 __global__ void haversine_distance_kernel(int pnt_size, const T* const __restrict__ x1,const T* const __restrict__ y1,
	const T* const __restrict__ x2,const T* const __restrict__ y2, T* const __restrict__ h_dist)
{
    //assuming 1D grid/block config
    uint32_t idx =blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=pnt_size) return;  
    T x_1 = pi/180 * x1[idx];
    T y_1 = pi/180 * y1[idx];
    T x_2 = pi/180 * x2[idx];
    T y_2 = pi/180 * y2[idx];
    T dlon = x_2 - x_1;
    T dlat = y_2 - y_1;
    T a = sin(dlat/2)*sin(dlat/2) + cos(y_1) * cos(y_2) * sin(dlon/2)*sin(dlon/2);
    T c = 2 * asin(sqrt(a));
    h_dist[idx]=c*6371;
}

struct haversine_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
     gdf_column operator()(const gdf_column& x1,const gdf_column& y1,const gdf_column& x2,const gdf_column& y2)
    				
    {
        gdf_column h_dist;
        col_type* data;
        
        int num_print=(x1.size<10)?x1.size:10;
        std::cout<<"showing the first "<< num_print<<" output records"<<std::endl;
        thrust::device_ptr<col_type> x1_ptr=thrust::device_pointer_cast(static_cast<col_type*>(x1.data));
        thrust::device_ptr<col_type> y1_ptr=thrust::device_pointer_cast(static_cast<col_type*>(y1.data));
        thrust::device_ptr<col_type> x2_ptr=thrust::device_pointer_cast(static_cast<col_type*>(x2.data));
        thrust::device_ptr<col_type> y2_ptr=thrust::device_pointer_cast(static_cast<col_type*>(y2.data));
        std::cout<<"x1:"<<std::endl;
        thrust::copy(x1_ptr,x1_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
        std::cout<<"y1:"<<std::endl;
  	thrust::copy(y1_ptr,y1_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
  	std::cout<<"x2:"<<std::endl;
        thrust::copy(x2_ptr,x2_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;    
        std::cout<<"y2:"<<std::endl;
 	thrust::copy(y2_ptr,y2_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
 	
        RMM_TRY( RMM_ALLOC(&data, x1.size * sizeof(col_type), 0) );
        gdf_column_view(&h_dist, data, nullptr, x1.size, x1.dtype);

        struct timeval t0,t1;
        gettimeofday(&t0, nullptr);
        
        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, haversine_distance_kernel<col_type>) );
        cudf::util::cuda::grid_config_1d grid{x1.size, block_size, 1};
        
        std::cout<<"num_points="<<x1.size<<" min_grid_size="<<min_grid_size<<" block_size="<<block_size<<std::endl;
        
        haversine_distance_kernel<col_type> <<< grid.num_blocks, block_size >>> (x1.size,
               	static_cast<col_type*>(x1.data), static_cast<col_type*>(y1.data),
        	static_cast<col_type*>(x2.data), static_cast<col_type*>(y2.data),
                static_cast<col_type*>(data) );
        CUDA_TRY( cudaDeviceSynchronize() );
	
	gettimeofday(&t1, nullptr);	
 	float haversine_distance_kernel_time=calc_time("haversine_distance_kernel_time in ms=",t0,t1);
        
        std::cout<<"haversine distance:"<<std::endl;
        thrust::device_ptr<col_type> d_hdist_ptr=thrust::device_pointer_cast(static_cast<col_type*>(data));
        thrust::copy(d_hdist_ptr,d_hdist_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
 
        return h_dist;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    gdf_column operator()(const gdf_column& x1,const gdf_column& y1,const gdf_column& x2,const gdf_column& y2)      				
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};

/**
 *@brief Compute Haversine distances among pairs of logitude/latitude locations
 *see haversine.hpp
*/

namespace cuspatial{

/**
 * @brief Compute Haversine distances among pairs of logitude/latitude locations
 * see haversine.hpp
*/

gdf_column haversine_distance(const gdf_column& x1,const gdf_column& y1,const gdf_column& x2,const gdf_column& y2 )                        
{       
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);
    
    CUDF_EXPECTS(x1.data != nullptr && y1.data != nullptr && x2.data != nullptr && y2.data != nullptr,"point lon/lat cannot be empty");
    CUDF_EXPECTS(x1.dtype == x2.dtype && x2.dtype==y1.dtype && y1.dtype==y2.dtype, "x1/x2/y1/y2 type mismatch");
    CUDF_EXPECTS(x1.size == x2.size && x2.size==y1.size && y1.size==y2.size, "x1/x2/y1/y2 size mismatch");
       
    //future versions might allow pnt_(x/y) have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(x1.null_count == 0 && y1.null_count == 0 && x2.null_count == 0 && y2.null_count == 0, "this version does not support x1/x2/y1/y2 contains nulls");
    
    gdf_column h_d = cudf::type_dispatcher( x1.dtype, haversine_functor(), x1,y1,x2,y2);
    		
    gettimeofday(&t1, nullptr);
    float haversine_end2end_time=calc_time("C++ haversine_distance end-to-end time in ms=",t0,t1);
    
    return h_d;
    
  }//haversine_distance 
  
}// namespace cuspatial
