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

#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <vector>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

#include <utility/helper_thrust.cuh>
#include <utility/quadtree_thrust.cuh>
#include <utility/bbox_thrust.cuh>
#include <cuspatial/bounding_box.hpp>
#include <cuspatial/spatial_jion.hpp>


namespace
{

const uint8_t max_warps_per_block=32;
const uint8_t num_threads_per_warp=32;

template <typename T>
__global__ void quad_pip_phase1_kernel(const uint32_t * pq_poly_id,const uint32_t *pq_quad_id,
	const uint32_t *pnt_length,const uint32_t *pnt_fpos, const T*  pnt_x,const T*  pnt_y, 
	const uint32_t*  poly_fpos,const uint32_t*  poly_rpos,const T*  poly_x,const T*  poly_y,
        uint32_t* num_hits)        
{
    __shared__ uint32_t qid,pid,num_point,first_pos,num_adjusted;
    
    //assume #of points/threads no more than num_threads_per_warp*max_warps_per_block (32*32)
    __shared__ uint32_t data[max_warps_per_block];
    //assuming 1d 
    if(threadIdx.x==0)
    {
    	qid=pq_quad_id[blockIdx.x];
    	pid=pq_poly_id[blockIdx.x];
    	num_point=pnt_length[qid];
    	first_pos=pnt_fpos[qid]; 
    	num_adjusted=((num_point-1)/num_threads_per_warp+1)*num_threads_per_warp;
       	//printf("block=%d qid=%d pid=%d num_point=%d first_pos=%d\n",
    	//	blockIdx.x,qid,pid,num_point,first_pos);	
    }
     __syncthreads();
     
    if((threadIdx.x>=max_warps_per_block)&&(threadIdx.x>=num_adjusted))
    	return;
    __syncthreads();
    
    if(threadIdx.x<max_warps_per_block)
        data[threadIdx.x]=0;
    __syncthreads();
    
    uint32_t tid = first_pos+threadIdx.x;
    bool in_polygon = false;
    if(threadIdx.x<num_point)
    {
       T x = pnt_x[tid];
       T y = pnt_y[tid];
      
       uint32_t r_f = (0 == pid) ? 0 : poly_fpos[pid-1];
       uint32_t r_t=poly_fpos[pid];
       for (uint32_t k = r_f; k < r_t; k++) //for each ring
       {
           uint32_t m = (k==0)?0:poly_rpos[k-1];
           for (;m < poly_rpos[k]-1; m++) //for each line segment
           {
              T x0, x1, y0, y1;
              x0 = poly_x[m];
              y0 = poly_y[m];
              x1 = poly_x[m+1];
              y1 = poly_y[m+1];
              //printf("block=%2d thread=%2d tid=%2d r_f=%2d r_t=%2d x=%10.5f y=%10.5f x0=%10.5f y0=%10.5f x1=%10.5f y1=%10.5f\n",
              //	blockIdx.x,threadIdx.x,tid,r_f,r_t,x,y,x0,y0,x1,y1);

              if ((((y0 <= y) && (y < y1)) ||
                   ((y1 <= y) && (y < y0))) &&
                       (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0))
                 in_polygon = !in_polygon;
            }//m
         }//k
      }
      __syncthreads();

      unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < num_point);
      uint32_t vote=__ballot_sync(mask,in_polygon);
      //printf("p1: block=%d thread=%d tid=%d in_polygon=%d mask=%08x vote=%08x\n",blockIdx.x,threadIdx.x,tid,in_polygon,mask,vote);
      
      if(threadIdx.x%num_threads_per_warp==0)
      	data[threadIdx.x/num_threads_per_warp]=__popc(vote);  
      __syncthreads();
      
      /*if(threadIdx.x<max_warps_per_block)
      	printf("p1: block=%d thread=%d data=%d\n",blockIdx.x,threadIdx.x,data[threadIdx.x]);
      __syncthreads();*/
      
      if(threadIdx.x<max_warps_per_block)
      {
      	uint32_t num=data[threadIdx.x];
        for (uint32_t offset = max_warps_per_block/2; offset > 0; offset /= 2) 
            num += __shfl_xor_sync(0xFFFFFFFF,num, offset);  	
        if(threadIdx.x==0)
            num_hits[blockIdx.x]=num;
      }
      __syncthreads();
}

template <typename T>
__global__ void quad_pip_phase2_kernel(const uint32_t * pq_poly_id,const uint32_t *pq_quad_id,
	const uint32_t *pnt_length,const uint32_t *pnt_fpos, const T*  pnt_x,const T*  pnt_y, const uint32_t* pnt_id,
	const uint32_t *poly_id,const uint32_t* poly_fpos,const uint32_t*  poly_rpos,const T*  poly_x,const T*  poly_y,
        uint32_t *d_num_hits,uint32_t *d_res_poly_id,uint32_t *d_res_pnt_id)        
{
    __shared__ uint32_t qid,pid,num_point,first_pos,mem_offset,num_adjusted;
    
    //assume #of points/threads no more than num_threads_per_warp*max_warps_per_block (32*32)
    __shared__ uint16_t temp[max_warps_per_block],sums[max_warps_per_block+1];

    //assuming 1d 
    if(threadIdx.x==0)
    {
    	qid=pq_quad_id[blockIdx.x];
    	pid=pq_poly_id[blockIdx.x];
    	num_point=pnt_length[qid];
    	first_pos=pnt_fpos[qid]; 
    	mem_offset=d_num_hits[blockIdx.x];
    	sums[0]=0;
    	num_adjusted=((num_point-1)/num_threads_per_warp+1)*num_threads_per_warp;
     	//printf("block=%d qid=%d pid=%d num_point=%d first_pos=%d mem_offset=%d\n",
    	//	blockIdx.x,qid,pid,num_point,first_pos,mem_offset);
    		
    }
    __syncthreads();

     if(threadIdx.x<max_warps_per_block+1)
    	temp[threadIdx.x]=0;
    __syncthreads();
   
    uint32_t tid = first_pos+threadIdx.x;    	
    bool in_polygon = false;
    if(threadIdx.x<num_point)
    {   
       T x = pnt_x[tid];
       T y = pnt_y[tid];
     
       uint32_t r_f = (0 == pid) ? 0 : poly_fpos[pid-1];
       uint32_t r_t=poly_fpos[pid];
       for (uint16_t k = r_f; k < r_t; k++) //for each ring
       {
           uint16_t m = (k==0)?0:poly_rpos[k-1];
           for (;m < poly_rpos[k]-1; m++) //for each line segment
           {
              T x0, x1, y0, y1;
              x0 = poly_x[m];
              y0 = poly_y[m];
              x1 = poly_x[m+1];
              y1 = poly_y[m+1];

              if ((((y0 <= y) && (y < y1)) ||
                   ((y1 <= y) && (y < y0))) &&
                       (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0))
                 in_polygon = !in_polygon;
            }//m
          }//k
      }
      __syncthreads();    
  
      unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < num_adjusted);
      uint32_t vote=__ballot_sync(mask,in_polygon);    
      if(threadIdx.x%num_threads_per_warp==0)
      	temp[threadIdx.x/num_threads_per_warp]=__popc(vote);  
      __syncthreads();
    
     //warp-level scan; only one warp is used
     if(threadIdx.x<num_threads_per_warp)
      {
          uint16_t num=temp[threadIdx.x];
          for (uint8_t i=1; i<=num_threads_per_warp; i*=2)
          {
            int n = __shfl_up_sync(0xFFFFFFF,num, i, num_threads_per_warp);
            if (threadIdx.x >= i) num += n;
          }
          sums[threadIdx.x+1]=num;
          __syncthreads();
      }
      //important!!!!!!!!!!!
      __syncthreads();
      
      /*if(threadIdx.x<num_point)
      	printf("after: block=%d thread=%d tid=%d %10.5f %10.5f in_polygon=%d val=%d\n",
      		blockIdx.x,threadIdx.x,tid,pnt_x[tid],pnt_y[tid],in_polygon,sums[threadIdx.x]);
      __syncthreads();*/
      
      if((threadIdx.x<num_point)&&(in_polygon))
      {
     	uint16_t num=sums[threadIdx.x/num_threads_per_warp];
     	uint16_t warp_offset=__popc(vote>>(threadIdx.x%num_threads_per_warp))-1;
     	uint16_t pos=num+warp_offset;
     	
     	//printf("block=%d thread=%d qid=%d pid=%d tid=%d mem_offset=%d num=%d warp_offset=%d pos=%d\n",
    	//	blockIdx.x,threadIdx.x,qid,pid,tid,mem_offset,num,warp_offset,pos); 
    		
        d_res_poly_id[mem_offset+pos]=poly_id[pid];
        d_res_pnt_id[mem_offset+pos]=pnt_id[tid];
      } 
      __syncthreads();
}

template<typename T>
std::vector<std::unique_ptr<cudf::column>> dowork(
	uint32_t num_pq_pair,const uint32_t * d_pq_polyid,const uint32_t * d_pq_quadid,
	uint32_t num_node,const uint32_t *d_qt_key,const uint8_t *d_qt_lev,
	const bool *d_qt_sign, const uint32_t *d_qt_length, const uint32_t *d_qt_fpos,
	const uint32_t num_pnt,const uint32_t * d_pnt_id,const T *d_pnt_x, const T *d_pnt_y,
	const uint32_t num_poly,const uint32_t * d_poly_id,const uint32_t * d_poly_fpos,
	const uint32_t * d_poly_rpos,const T *d_poly_x, const T *d_poly_y,
	rmm::mr::device_memory_resource* mr, cudaStream_t stream)	
                                         
{
    auto exec_policy = rmm::exec_policy(stream)->on(stream);
    
    uint32_t *d_num_hits=NULL;
    RMM_TRY( RMM_ALLOC( &d_num_hits,num_pq_pair* sizeof(uint32_t), 0));
    assert(d_num_hits!=NULL);
    HANDLE_CUDA_ERROR( cudaMemset(d_num_hits,0,num_pq_pair*sizeof(uint32_t)) ); 

    printf("running quad_pip_phase1_kernel\n");
    quad_pip_phase1_kernel<T> <<< num_pq_pair, 256 >>> (d_pq_polyid,d_pq_quadid,
    	d_qt_length,d_qt_fpos,d_pnt_x,d_pnt_y,d_poly_fpos,d_poly_rpos,d_poly_x,d_poly_y,d_num_hits);
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );	

    //remove poly-quad pair with zero hits
 
    uint32_t *d_tmp_polyid=NULL,*d_tmp_quadid=NULL;
    RMM_TRY( RMM_ALLOC( &d_tmp_polyid,num_pq_pair* sizeof(uint32_t), stream));
    RMM_TRY( RMM_ALLOC( &d_tmp_quadid,num_pq_pair* sizeof(uint32_t), stream));
    assert(d_tmp_polyid!=NULL && d_tmp_quadid!=NULL);
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_tmp_polyid, (void *)d_pq_polyid, num_pq_pair * sizeof(uint32_t), cudaMemcpyDeviceToDevice ) );
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_tmp_quadid, (void *)d_pq_quadid, num_pq_pair * sizeof(uint32_t), cudaMemcpyDeviceToDevice ) );

if(1)
{
	printf("phase1 results before remove:\n");	
	thrust::device_ptr<uint32_t> d_num_hits_ptr=thrust::device_pointer_cast(d_num_hits);		
	printf("d_num_hits: before reduce\n");
        thrust::copy(d_num_hits_ptr,d_num_hits_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
}
    auto valid_pq_pair_iter=thrust::make_zip_iterator(thrust::make_tuple(d_tmp_polyid,d_tmp_quadid,d_num_hits));    
    uint32_t num_valid_pair=thrust::remove_if(exec_policy,valid_pq_pair_iter,valid_pq_pair_iter+num_pq_pair,
    	valid_pq_pair_iter,pq_remove_zero())-valid_pq_pair_iter;   
    printf("num_valid_pair=%d\n",num_valid_pair);

if(1)
{
	printf("phase1 results after remove:\n");	
	thrust::device_ptr<uint32_t> d_num_hits_ptr=thrust::device_pointer_cast(d_num_hits);		
	printf("d_num_hits: before reduce\n");
        thrust::copy(d_num_hits_ptr,d_num_hits_ptr+num_valid_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
}    
    uint32_t total_hits=thrust::reduce(exec_policy,d_num_hits,d_num_hits+num_valid_pair);
    printf("total_hits=%d\n",total_hits);
    thrust::exclusive_scan(exec_policy,d_num_hits,d_num_hits+num_valid_pair,d_num_hits);

if(1)
{
	printf("phase1 results:\n");	
	thrust::device_ptr<uint32_t> d_num_hits_ptr=thrust::device_pointer_cast(d_num_hits);		
	printf("d_num_hits: after reduce\n");
        thrust::copy(d_num_hits_ptr,d_num_hits_ptr+num_valid_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
}
  
    std::unique_ptr<cudf::column> poly_id_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), total_hits,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint32_t *d_res_poly_id=cudf::mutable_column_device_view::create(poly_id_col->mutable_view(), stream)->data<uint32_t>();
    CUDF_EXPECTS(d_res_poly_id!=NULL,"poly_id can not be NULL"); 
   
    std::unique_ptr<cudf::column> pnt_id_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), total_hits,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint32_t *d_res_pnt_id=cudf::mutable_column_device_view::create(pnt_id_col->mutable_view(), stream)->data<uint32_t>();
    CUDF_EXPECTS(d_res_pnt_id!=NULL,"point_id can not be NULL"); 
        
    printf("running quad_pip_phase2_kernel\n");
    quad_pip_phase2_kernel<T> <<< num_valid_pair, 256 >>> 
       (d_tmp_polyid,d_tmp_quadid,
    	d_qt_length,d_qt_fpos,d_pnt_x,d_pnt_y,d_pnt_id,
    	d_poly_id,d_poly_fpos,d_poly_rpos,d_poly_x,d_poly_y,
    	d_num_hits,d_res_poly_id,d_res_pnt_id);   
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    
    RMM_TRY(RMM_FREE(d_num_hits,stream));d_num_hits=NULL;
    RMM_TRY(RMM_FREE(d_tmp_polyid,stream));d_tmp_polyid=NULL;
    RMM_TRY(RMM_FREE(d_tmp_quadid,stream));d_tmp_quadid=NULL;
    
   //thrust::stable_sort_by_key(exec_policy,d_res_pnt_id,d_res_pnt_id+total_hits,d_res_poly_id);
   
if(1)
{
	printf("phase2 results:\n");	

	thrust::device_ptr<uint32_t> d_res_poly_ptr=thrust::device_pointer_cast(d_res_poly_id);		
	printf("d_res_poly_id\n");
        thrust::copy(d_res_poly_ptr,d_res_poly_ptr+total_hits,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 

	thrust::device_ptr<uint32_t> d_res_pnt_ptr=thrust::device_pointer_cast(d_res_pnt_id);		
	printf("d_res_pnt_id\n");
        thrust::copy(d_res_pnt_ptr,d_res_pnt_ptr+total_hits,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
}

   std::vector<std::unique_ptr<cudf::column>> pair_cols;
   pair_cols.push_back(std::move(poly_id_col));
   pair_cols.push_back(std::move(pnt_id_col));
   return pair_cols;    
}

struct pip_refine_processor {
  
  template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(
	cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
	cudf::column_view const& poly_id,cudf::column_view const& poly_fpos,cudf::column_view const& poly_rpos,
	cudf::column_view const& poly_x,cudf::column_view const& poly_y,
	rmm::mr::device_memory_resource* mr,
        cudaStream_t stream)
   {            
       const uint32_t *d_poly_id=poly_id.data<uint32_t>();       
       const uint32_t *d_poly_fpos=poly_fpos.data<uint32_t>();
       const uint32_t *d_poly_rpos=poly_rpos.data<uint32_t>();
       const T *d_poly_x=poly_x.data<T>();
       const T *d_poly_y=poly_y.data<T>();
  
       const uint32_t *d_pnt_id=pnt.column(0).data<uint32_t>();       
       const T *d_pnt_x=pnt.column(1).data<T>();
       const T *d_pnt_y=pnt.column(2).data<T>();

       const uint32_t *d_qt_key=    quadtree.column(0).data<uint32_t>();
       const uint8_t  *d_qt_lev=    quadtree.column(1).data<uint8_t>();
       const bool     *d_qt_sign=   quadtree.column(2).data<bool>();
       const uint32_t *d_qt_length= quadtree.column(3).data<uint32_t>();
       const uint32_t *d_qt_fpos=   quadtree.column(4).data<uint32_t>();
       
       //column 0 and 1 of table pq_pair are for debug use only
       const uint32_t *d_pq_poly_id=   pq_pair.column(2).data<uint32_t>();
       const uint32_t *d_pq_quad_id=   pq_pair.column(3).data<uint32_t>();
             
       uint32_t num_pair=pq_pair.num_rows();
       uint32_t num_node=quadtree.num_rows();
       uint32_t num_poly=poly_id.size();
       uint32_t num_pnt=pnt.num_rows();

       std::vector<std::unique_ptr<cudf::column>> pair_cols=
       		dowork(num_pair,d_pq_poly_id,d_pq_quad_id,
       			num_node,d_qt_key,d_qt_lev,d_qt_sign,d_qt_length,d_qt_fpos,
       			num_pnt,d_pnt_id,d_pnt_x,d_pnt_y,
       			num_poly,d_poly_id,d_poly_fpos,d_poly_rpos,d_poly_x,d_poly_y,
       			mr,stream);
       	
      std::unique_ptr<cudf::experimental::table> destination_table = 
    	std::make_unique<cudf::experimental::table>(std::move(pair_cols));      
      
      return destination_table;
    }
  
  template<typename T, std::enable_if_t<!std::is_floating_point<T>::value >* = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(
	cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
	cudf::column_view const& poly_id,cudf::column_view const& poly_fpos,cudf::column_view const& poly_rpos,
	cudf::column_view const& poly_x,cudf::column_view const& poly_y,
	rmm::mr::device_memory_resource* mr, cudaStream_t stream)       
    {
 	CUDF_FAIL("Non-floating point operation is not supported");
    }  
      
};
  
} //end anonymous namespace

namespace cuspatial {
std::unique_ptr<cudf::experimental::table> pip_refine(
	cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
	cudf::column_view const& poly_id,cudf::column_view const& poly_fpos,cudf::column_view const& poly_rpos,
	cudf::column_view const& poly_x,cudf::column_view const& poly_y)
	
	
{   
   cudf::data_type pnt_dtype=pnt.column(1).type();
   cudf::data_type poly_dtype=poly_x.type();
   CUDF_EXPECTS(pnt_dtype==poly_dtype,"point and polygon must have the same data type");
   
   cudaStream_t stream=0;
   rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();    
   
   return cudf::experimental::type_dispatcher(pnt_dtype,pip_refine_processor{}, 
    	pq_pair,quadtree,pnt,poly_id,poly_fpos,poly_rpos,poly_x,poly_y,mr,stream);   
    	    	
}

}// namespace cuspatial
