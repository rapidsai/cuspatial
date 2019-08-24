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
__global__ void kernel_Hausdorff_Pair(
                int num_pair,
                uint32_t *seg_left,
                uint32_t *seg_right,
                T *xx,
                T *yy,
                uint32_t *pos,
                T *results
                )
{
    int bidx = blockIdx.y*gridDim.x+blockIdx.x;
    if (bidx < num_pair)
    {
        int seg_id_left = seg_left[bidx];
        int seg_id_right = seg_right[bidx];
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
        /*if(bidx<100 && threadIdx.x==0)
        	printf("(%d %d %d) (%d %d) (%d,%d)\n",bidx,seg_id_left,seg_id_right, start_left,stop_left,start_right,stop_right);*/
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
    //if(bidx<100 && threadIdx.x == 0) printf("%d %10.5f\n",bidx,results[bidx]);
}


template <typename T>
__global__ void kernel_Hausdorff_Sym(int num_set,T *in,T *out )                               
{
    int tidx = (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tidx<num_set*num_set)
    {
    	int indx=(tidx%num_set)*num_set+(tidx/num_set);
    	out[tidx]=(in[tidx]>in[indx])?in[tidx]:in[indx];
    	out[indx]=out[tidx];
    }
}


struct Hausdorff_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
         return std::is_floating_point<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    gdf_column  operator()(const gdf_column& coord_x,const gdf_column& coord_y,const gdf_column& cnt
    		/* ,cudaStream_t stream = 0   */)
    	
    { 
 	gdf_column d_matrix;
 	int num_set=cnt.size;
  	int block_sz = num_set*num_set;
 	d_matrix.dtype= coord_x.dtype;
  	d_matrix.col_name=(char *)malloc(strlen("dist")+ 1);
	strcpy(d_matrix.col_name,"dist");    
        RMM_TRY( RMM_ALLOC(&d_matrix.data, block_sz * sizeof(col_type), 0) );
     	d_matrix.size=block_sz;
     	d_matrix.valid=nullptr;
     	d_matrix.null_count=0;		
        
        struct timeval t0,t1;
        gettimeofday(&t0, NULL);
     
        uint32_t *d_pos=NULL;
        RMM_TRY( RMM_ALLOC((void**)&d_pos, sizeof(uint32_t)*num_set, 0) );
        thrust::device_ptr<uint32_t> trajcnt_ptr=thrust::device_pointer_cast(static_cast<uint32_t*>(cnt.data));
        thrust::device_ptr<uint32_t> trajpos_ptr=thrust::device_pointer_cast(d_pos);
        thrust::inclusive_scan(trajcnt_ptr,trajcnt_ptr+num_set,trajpos_ptr);
        col_type *d_tempdis=NULL;
        RMM_TRY( RMM_ALLOC((void**)&d_tempdis, sizeof(col_type)*block_sz, 0) )
        assert(d_tempdis!=NULL);
         
        int block_x = block_sz, block_y = 1;
        if (block_sz > 65535)
        {
    	    block_y = ceil((float)block_sz/65535.0);
    	    block_x = 65535;
    	}
    	printf("block_sz=%d  block: %d - %d\n", block_sz,block_x, block_y);
    	
    	dim3 grid(block_x, block_y);
    	dim3 block(NUM_THREADS);   
        /*kernel_Hausdorff_Pair<col_type> <<< grid,block >>> (block_sz,d_pairs_left,d_pairs_right,        	
         	static_cast<col_type*>(coord_x.data),static_cast<col_type*>(coord_y.data),
        	d_pos,static_cast<col_type*>(d_matrix.data));*/
 
 	kernel_Hausdorff_Full<col_type> <<< grid,block >>> (num_set,        	
          	static_cast<col_type*>(coord_x.data),static_cast<col_type*>(coord_y.data),
         	d_pos,static_cast<col_type*>(d_matrix.data));
 
 
 	/*kernel_Hausdorff_Full<col_type> <<< grid,block >>> (num_set,        	
         	static_cast<col_type*>(coord_x.data),static_cast<col_type*>(coord_y.data),
        	d_pos,d_tempdis);
        kernel_Hausdorff_Sym<<< grid,block >>> (num_set,d_tempdis,static_cast<col_type*>(d_matrix.data));*/
        
         
        CUDA_TRY( cudaDeviceSynchronize() );
	gettimeofday(&t1, NULL);
	float kernelexec_time=calc_time("kernel exec_time:",t0,t1);
        //CHECK_STREAM(stream);        
        RMM_TRY( RMM_FREE(d_pos, 0) );
        //RMM_TRY( RMM_FREE(d_tempdis, 0) );
        
        int num_print=(d_matrix.size<10)?d_matrix.size:10;
        std::cout<<"showing the first "<< num_print<<" output records"<<std::endl;
        thrust::device_ptr<col_type> dist_ptr=thrust::device_pointer_cast(static_cast<col_type*>(d_matrix.data));
        std::cout<<"distance:"<<std::endl;
        thrust::copy(dist_ptr,dist_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl; 
        return d_matrix;
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    gdf_column  operator()(const gdf_column& coord_x,const gdf_column& coord_y,const gdf_column& cnt
    		/* ,cudaStream_t stream = 0   */)
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};
    

/**
* @Brief compute Hausdorff distances among all pairs of a set of trajectories
* see hausdorff.hpp
*/

namespace cuspatial {

gdf_column directed_hausdorff_distance(const gdf_column& coord_x,const gdf_column& coord_y,const gdf_column& cnt
    		/* ,cudaStream_t stream = 0   */)
{       
    struct timeval t0,t1;
    gettimeofday(&t0, NULL);
    
    CUDF_EXPECTS(coord_x.data != nullptr &&coord_y.data!=nullptr && cnt.data!=NULL,
    	"coord_x/coord_y/cnt data can not be null");
    CUDF_EXPECTS(coord_x.size == coord_y.size ,"coord_x/coord_y/must have the same size");
     
    //future versions might allow coord_x/coord_y/cnt have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(coord_x.null_count == 0 && coord_y.null_count == 0 && cnt.null_count==0,
    	"this version does not support coord_x/coord_y/cnt contains nulls");
    
    CUDF_EXPECTS(coord_x.size >= cnt.size ,"one trajectory must have at least one point");
 
  
    gdf_column dist =cudf::type_dispatcher(coord_x.dtype, Hausdorff_functor(), coord_x,coord_y,cnt/*,stream */);
    
    gettimeofday(&t1, NULL);
    float Hausdorff_end2end_time=calc_time("C++ Hausdorff end-to-end time in ms=",t0,t1);
    return dist;
    
    }//hausdorff_distance     
    	
}// namespace cuspatial
