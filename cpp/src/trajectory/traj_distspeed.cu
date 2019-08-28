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

#include <cuspatial/shared_util.h>
#include <cuspatial/traj_thrust.h>
#include <cuspatial/trajectory.hpp>

using namespace std; 
using namespace cudf;
using namespace cuspatial;

template <typename T>
__global__ void distspeed_kernel(gdf_size_type num_traj,const T* const __restrict__ coord_x,const T* const __restrict__ coord_y,
	 const its_timestamp *const __restrict__ time,const uint32_t * const __restrict__ len,const uint32_t * const __restrict__ pos,
	 T* const __restrict__ dis, T* const __restrict__ sp)
	 
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
   	 		float dt=(coord_x[bp+i+1]-coord_x[bp+i])*(coord_x[bp+i+1]-coord_x[bp+i]);
   	 		dt+=(coord_y[bp+i+1]-coord_y[bp+i])*(coord_y[bp+i+1]-coord_y[bp+i]);
   	 		ds+=sqrt(dt);
   	 	}
   	 	dis[pid]=ds*1000; //km to m
   	 	sp[pid]=ds*1000/td; // m/s
   	 }
}

struct distspeed_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
         return std::is_floating_point<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    void operator()(const gdf_column& coord_x,const gdf_column& coord_y,const gdf_column& ts,
 			    const gdf_column& len,const gdf_column& pos,
 			    gdf_column& dist,gdf_column& speed)
    	
    { 
 	dist.dtype= coord_x.dtype;
  	dist.col_name=(char *)malloc(strlen("dist")+ 1);
	strcpy(dist.col_name,"dist");    
        RMM_TRY( RMM_ALLOC(&dist.data, len.size * sizeof(col_type), 0) );
     	dist.size=len.size;
     	dist.valid=nullptr;
     	dist.null_count=0;		

 	speed.dtype= coord_x.dtype;
  	speed.col_name=(char *)malloc(strlen("speed")+ 1);
	strcpy(dist.col_name,"speed");    
        RMM_TRY( RMM_ALLOC(&speed.data, len.size * sizeof(col_type), 0) );
     	speed.size=len.size;
     	speed.valid=nullptr;
     	speed.null_count=0;	
        
        struct timeval t0,t1;
        gettimeofday(&t0, nullptr);
        
        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, distspeed_kernel<col_type>) );
        cudf::util::cuda::grid_config_1d grid{coord_x.size, block_size, 1};
        std::cout<<"coord_x.size="<<coord_x.size<<" block_size="<<block_size<<std::endl;
       
        distspeed_kernel<col_type> <<< grid.num_blocks, block_size >>> (len.size,
        	static_cast<col_type*>(coord_x.data),static_cast<col_type*>(coord_y.data),
        	static_cast<its_timestamp*>(ts.data),static_cast<uint32_t*>(len.data), static_cast<uint32_t*>(pos.data),
   	    	static_cast<col_type*>(dist.data), static_cast<col_type*>(speed.data) );           
        CUDA_TRY( cudaDeviceSynchronize() );

	gettimeofday(&t1, nullptr);
	float distspeed_kernel_time=calc_time("distspeed_kernel_time in ms=",t0,t1);
        //CHECK_STREAM(stream);
        
        int num_print=(len.size<10)?len.size:10;
        std::cout<<"showing the first "<< num_print<<" output records"<<std::endl;
        thrust::device_ptr<col_type> dist_ptr=thrust::device_pointer_cast(static_cast<col_type*>(dist.data));
        thrust::device_ptr<col_type> speed_ptr=thrust::device_pointer_cast(static_cast<col_type*>(speed.data));
        std::cout<<"distance:"<<std::endl;
        thrust::copy(dist_ptr,dist_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl; 
        std::cout<<"speed:"<<std::endl;
 	thrust::copy(speed_ptr,speed_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    void operator()(const gdf_column& coord_x,const gdf_column& coord_y,const gdf_column& ts,
 			    const gdf_column& len,const gdf_column& pos,
 			    gdf_column& dist,gdf_column& speed)
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};
    

/**
 * @Brief computing distance(length) and speed of trajectories after their formation (e.g., from coord_to_traj)
 * see trajectory.hpp
 */
 
namespace cuspatial {

void traj_distspeed(const gdf_column& coord_x,const gdf_column& coord_y,const gdf_column& ts,
 			    const gdf_column& len,const gdf_column& pos,gdf_column& dist,gdf_column& speed)
 			    
{       
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);
    
    CUDF_EXPECTS(coord_x.data != nullptr &&coord_y.data!=nullptr && ts.data!=nullptr && len.data!=nullptr && pos.data!=nullptr,
    	"coord_x/coord_y/ts/len/pos data can not be null");
    CUDF_EXPECTS(coord_x.size == coord_y.size && coord_x.size==ts.size ,"coord_x/coord_y/ts must have the same size");
    CUDF_EXPECTS(len.size == pos.size ,"len/pos must have the same size");
     
    //future versions might allow coord_x/coord_y/ts/pos/len have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(coord_x.null_count == 0 && coord_y.null_count == 0 && ts.null_count==0 && len.null_count==0 &&  pos.null_count==0,
    	"this version does not support coord_x/coord_y/ts/len/pos contains nulls");
    
    CUDF_EXPECTS(coord_x.size >= pos.size ,"one trajectory must have at least one point");
 
  
    cudf::type_dispatcher(coord_x.dtype, distspeed_functor(), coord_x,coord_y,ts,len,pos,dist,speed/*,stream */);
    
    gettimeofday(&t1, nullptr);
    float distspeed_end2end_time=calc_time("C++ traj_distspeed end-to-end time in ms=",t0,t1);
    
    }//traj_distspeed     
    	
}// namespace cuspatial