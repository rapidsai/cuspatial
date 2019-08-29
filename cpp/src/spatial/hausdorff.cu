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
#include <cuspatial/hausdorff.hpp>

using namespace std; 
using namespace cudf;
using namespace cuspatial;

const unsigned int NUM_THREADS = 1024;
 
template <typename T>
__global__ void kernel_Hausdorff_Full(
                int num_traj,
                T *xx,
                T *yy,
                uint32_t *pos,
                T *results
                )
{
    int bidx = blockIdx.y*gridDim.x+blockIdx.x;
    if (bidx < num_traj*num_traj)
    {
        int seg_id_left = bidx/num_traj;
        int seg_id_right =bidx%num_traj;
        __shared__ T sdata[NUM_THREADS];
        sdata[threadIdx.x] = -1;
        __syncthreads();
         int start_left = seg_id_left == 0 ? 0 : pos[seg_id_left-1];
        int stop_left = pos[seg_id_left];

        int start_right = seg_id_right == 0 ? 0 : pos[seg_id_right-1];
        int stop_right = pos[seg_id_right];
        T dist = 1e20;
        int max_threads = 0;
        {
            max_threads = stop_left-start_left;
            if (threadIdx.x < max_threads)
            {
                T my_xx = xx[start_left+threadIdx.x];
                T my_yy = yy[start_left+threadIdx.x];
                for (int i = start_right; i < stop_right; i++)
                {
                    T other_xx = xx[i];
                    T other_yy = yy[i];
                    T new_dist = (my_xx-other_xx)*(my_xx-other_xx)
                        + (my_yy-other_yy)*(my_yy-other_yy);
                    dist= min(dist, new_dist);//dist < new_dist ? dist : new_dist;
                }
            }
        }
        if (dist > 1e10)
            dist = -1;

         if(threadIdx.x < max_threads)
                   sdata[threadIdx.x] = dist;
        __syncthreads();
        //reduction
        for(int offset = blockDim.x / 2;
                offset > 0;
                offset >>= 1)
        {
            if(threadIdx.x < offset)
            {
                T tmp = sdata[threadIdx.x + offset];
                T tmp2 = sdata[threadIdx.x];
                sdata[threadIdx.x] = max(tmp2, tmp);
            }

            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == 0)
            results[bidx] = (sdata[0]>=0)?sqrt(sdata[0]):1e10;
    }
}

struct Hausdorff_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
         return std::is_floating_point<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    gdf_column  operator()(const gdf_column& x,const gdf_column& y,const gdf_column& vertex_counts)    		    	
    { 
 	gdf_column d_matrix;
 	int num_set=vertex_counts.size;
  	int block_sz = num_set*num_set;
 	d_matrix.dtype= x.dtype;
  	d_matrix.col_name=(char *)malloc(strlen("dist")+ 1);
	strcpy(d_matrix.col_name,"dist");    
        RMM_TRY( RMM_ALLOC(&d_matrix.data, block_sz * sizeof(col_type), 0) );
     	d_matrix.size=block_sz;
     	d_matrix.valid=nullptr;
     	d_matrix.null_count=0;		
        
        struct timeval t0,t1;
        gettimeofday(&t0, nullptr);
     
        uint32_t *d_pos=nullptr;
        RMM_TRY( RMM_ALLOC((void**)&d_pos, sizeof(uint32_t)*num_set, 0) );
        thrust::device_ptr<uint32_t> vertex_counts_ptr=thrust::device_pointer_cast(static_cast<uint32_t*>(vertex_counts.data));
        thrust::device_ptr<uint32_t> vertex_positions_ptr=thrust::device_pointer_cast(d_pos);
        thrust::inclusive_scan(vertex_counts_ptr,vertex_counts_ptr+num_set,vertex_positions_ptr);
        
        int block_x = block_sz, block_y = 1;
        if (block_sz > 65535)
        {
    	    block_y = ceil((float)block_sz/65535.0);
    	    block_x = 65535;
    	}
    	printf("block_sz=%d  block: %d - %d\n", block_sz,block_x, block_y);
    	
    	dim3 grid(block_x, block_y);
    	dim3 block(NUM_THREADS);   
 
 	kernel_Hausdorff_Full<col_type> <<< grid,block >>> (num_set,        	
          	static_cast<col_type*>(x.data),static_cast<col_type*>(y.data),
         	d_pos,static_cast<col_type*>(d_matrix.data));
     
         
        CUDA_TRY( cudaDeviceSynchronize() );
	gettimeofday(&t1, nullptr);
	float kernelexec_time=calc_time("kernel exec_time:",t0,t1);
        RMM_TRY( RMM_FREE(d_pos, 0) );
       
        int num_print=(d_matrix.size<10)?d_matrix.size:10;
        std::cout<<"showing the first "<< num_print<<" output records"<<std::endl;
        thrust::device_ptr<col_type> dist_ptr=thrust::device_pointer_cast(static_cast<col_type*>(d_matrix.data));
        std::cout<<"distance:"<<std::endl;
        thrust::copy(dist_ptr,dist_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl; 
        return d_matrix;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    gdf_column  operator()(const gdf_column& x,const gdf_column& y,const gdf_column& vertex_counts)
    		
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};
    

/**
* @brief compute Hausdorff distances among all pairs of a set of trajectories
* see hausdorff.hpp
*/

namespace cuspatial {

gdf_column directed_hausdorff_distance(const gdf_column& x,const gdf_column& y,const gdf_column& vertex_counts)
    		
{       
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);
    
    CUDF_EXPECTS(x.data != nullptr &&y.data!=nullptr && vertex_counts.data!=nullptr,
    	"x/y/vertex_counts data can not be null");
    CUDF_EXPECTS(x.size == y.size ,"x/y/must have the same size");
     
    //future versions might allow x/y/vertex_counts have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(x.null_count == 0 && y.null_count == 0 && vertex_counts.null_count==0,
    	"this version does not support x/y/vertex_counts contains nulls");
    
    CUDF_EXPECTS(x.size >= vertex_counts.size ,"one trajectory must have at least one point");
 
  
    gdf_column dist =cudf::type_dispatcher(x.dtype, Hausdorff_functor(), x,y,vertex_counts);
    
    gettimeofday(&t1, nullptr);
    float Hausdorff_end2end_time=calc_time("C++ Hausdorff end-to-end time in ms=",t0,t1);
    return dist;
    
    }//hausdorff_distance     
    	
}// namespace cuspatial
