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

#include <time.h>
#include <sys/time.h>

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

#include <utility/utility.hpp>
#include <utility/helper_thrust.cuh>
#include <utility/join_thrust.cuh>
#include <cuspatial/spatial_jion.hpp>


namespace
{

template <typename T>
__global__ void quad_pip_phase1_kernel(const uint32_t * pq_poly_id,const uint32_t *pq_quad_id,
    const uint32_t *block_offset,const uint32_t * block_length,
    const uint32_t *qt_fpos, const T*  pnt_x,const T*  pnt_y, 
    const uint32_t*  poly_fpos,const uint32_t*  poly_rpos,const T*  poly_x,const T*  poly_y,
    uint32_t* num_hits)
{
    __shared__ uint32_t qid,pid,num_point,first_pos,qpos,num_adjusted;

    //assume #of points/threads no more than num_threads_per_warp*max_warps_per_block (32*32)
    __shared__ uint16_t data[max_warps_per_block];

    //assuming 1d 
    if(threadIdx.x==0)
    {
        qid=pq_quad_id[blockIdx.x];
        pid=pq_poly_id[blockIdx.x];
        qpos=block_offset[blockIdx.x];
        num_point=block_length[blockIdx.x];
        first_pos=qt_fpos[qid]+qpos;  
        num_adjusted=((num_point-1)/num_threads_per_warp+1)*num_threads_per_warp;
        //printf("block=%d qid=%d pid=%d num_point=%d first_pos=%d\n",
        //blockIdx.x,qid,pid,num_point,first_pos);
    }
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
               //    blockIdx.x,threadIdx.x,tid,r_f,r_t,x,y,x0,y0,x1,y1);

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
    //printf("p1: block=%d thread=%d tid=%d in_polygon=%d mask=%08x vote=%08x\n",blockIdx.x,threadIdx.x,tid,in_polygon,mask,vote);

    if(threadIdx.x%num_threads_per_warp==0)
        data[threadIdx.x/num_threads_per_warp]=__popc(vote);
    __syncthreads();

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
    uint32_t *block_offset,uint32_t * block_length,
    const uint32_t *qt_fpos, const T*  pnt_x,const T*  pnt_y,
    const uint32_t*  poly_fpos,const uint32_t*  poly_rpos,const T*  poly_x,const T*  poly_y,
    uint32_t *d_num_hits,uint32_t *d_res_poly_id,uint32_t *d_res_pnt_id)
{
    __shared__ uint32_t qid,pid,num_point,first_pos,mem_offset,qpos,num_adjusted;

    //assume #of points/threads no more than num_threads_per_warp*max_warps_per_block (32*32)
    __shared__ uint16_t temp[max_warps_per_block],sums[max_warps_per_block+1];

    //assuming 1d 
    if(threadIdx.x==0)
    {
        qid=pq_quad_id[blockIdx.x];
        pid=pq_poly_id[blockIdx.x];
        qpos=block_offset[blockIdx.x];
        num_point=block_length[blockIdx.x];
        mem_offset=d_num_hits[blockIdx.x];
        first_pos=qt_fpos[qid]+qpos; 
        sums[0]=0;
        num_adjusted=((num_point-1)/num_threads_per_warp+1)*num_threads_per_warp;
        //printf("block=%d qid=%d pid=%d num_point=%d first_pos=%d mem_offset=%d\n",
        //    blockIdx.x,qid,pid,num_point,first_pos,mem_offset);
        
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
        uint32_t num=temp[threadIdx.x];
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

    if((threadIdx.x<num_point)&&(in_polygon))
    {
        uint16_t num=sums[threadIdx.x/num_threads_per_warp];
        uint16_t warp_offset=__popc(vote>>(threadIdx.x%num_threads_per_warp))-1;
        uint32_t pos=mem_offset+num+warp_offset;

        //printf("block=%d thread=%d qid=%d pid=%d tid=%d mem_offset=%d num=%d warp_offset=%d pos=%d\n",
        //    blockIdx.x,threadIdx.x,qid,pid,tid,mem_offset,num,warp_offset,pos);

        d_res_poly_id[pos]=pid;
        d_res_pnt_id[pos]=tid;
    }
    __syncthreads();
}

template<typename T>
std::vector<std::unique_ptr<cudf::column>> dowork(
    uint32_t num_org_pair,const uint32_t * d_org_poly_idx,const uint32_t * d_org_quad_idx,
    uint32_t num_node,const uint32_t *d_qt_key,const uint8_t *d_qt_lev,
    const bool *d_qt_sign, const uint32_t *d_qt_length, const uint32_t *d_qt_fpos,
    const uint32_t num_pnt,const T *d_pnt_x, const T *d_pnt_y,
    const uint32_t num_poly,const uint32_t * d_poly_fpos,
    const uint32_t * d_poly_rpos,const T *d_poly_x, const T *d_poly_y,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
                                         
{
    auto exec_policy = rmm::exec_policy(stream)->on(stream);

    //compute the total number of sub-pairs (units) using transform_reduce
    uint32_t num_pq_pair=thrust::transform_reduce(exec_policy,d_org_quad_idx,
        d_org_quad_idx+num_org_pair,get_num_units(d_qt_length,threads_per_block),0, thrust::plus<uint32_t>());
     std::cout<<"num_pq_pair="<<num_pq_pair<<std::endl;

    //allocate memory for both numbers and their prefix-sums
    rmm::device_buffer *db_num_units = new rmm::device_buffer(num_org_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_num_units!=nullptr, "Error allocating memory for array of numbers of sub-pairs (units)");
    uint32_t *d_num_units=static_cast<uint32_t *>(db_num_units->data());

    rmm::device_buffer *db_num_sums = new rmm::device_buffer(num_org_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_num_sums!=nullptr, "Error allocating memory for array of offsets ");
    uint32_t *d_num_sums=static_cast<uint32_t *>(db_num_sums->data());

    //computes numbers of sub-pairs for each quadrant-polygon pairs
    thrust::transform(exec_policy,d_org_quad_idx,d_org_quad_idx+num_org_pair,
        d_num_units,get_num_units(d_qt_length,threads_per_block));

if(0)
{
    std::cout<<"preprocess: d_org_poly_id"<<std::endl;

    thrust::device_ptr<const uint32_t> d_org_poly_idx_ptr=thrust::device_pointer_cast(d_org_poly_idx);
    thrust::copy(d_org_poly_idx_ptr,d_org_poly_idx_ptr+num_org_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    thrust::device_ptr<const uint32_t> d_org_quad_idx_ptr=thrust::device_pointer_cast(d_org_quad_idx);
    thrust::copy(d_org_quad_idx_ptr,d_org_quad_idx_ptr+num_org_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"preprocess: d_num_units"<<std::endl;
    thrust::device_ptr<uint32_t> d_num_units_ptr=thrust::device_pointer_cast(d_num_units);
    thrust::copy(d_num_units_ptr,d_num_units_ptr+num_org_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

    thrust::exclusive_scan(exec_policy,d_num_units,d_num_units+num_org_pair,d_num_sums);

if(0)
{
    std::cout<<"preprocess: d_num_sums"<<std::endl;
    thrust::device_ptr<uint32_t> d_num_sum_ptr=thrust::device_pointer_cast(d_num_sums);
    thrust::copy(d_num_sum_ptr,d_num_sum_ptr+num_org_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

   //allocate memory for sub-pairs with four components: (polygon_idx, quadrant_idx, offset, length)

    rmm::device_buffer *db_pq_poly_idx = new rmm::device_buffer(num_pq_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_pq_poly_idx!=nullptr, "Error allocating memory for array of polygon idx)");
    uint32_t *d_pq_poly_idx=static_cast<uint32_t *>(db_pq_poly_idx->data());

    rmm::device_buffer *db_pq_quad_idx = new rmm::device_buffer(num_pq_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_pq_quad_idx!=nullptr, "Error allocating memory for array of quadrant idx)");
    uint32_t *d_pq_quad_idx=static_cast<uint32_t *>(db_pq_quad_idx->data());

    rmm::device_buffer *db_quad_offset = new rmm::device_buffer(num_pq_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_quad_offset!=nullptr, "Error allocating memory for array of sub-pair offsets )");
    uint32_t *d_quad_offset=static_cast<uint32_t *>(db_quad_offset->data());

    rmm::device_buffer *db_quad_len = new rmm::device_buffer(num_pq_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_quad_len!=nullptr, "Error allocating memory for array of sub-pair length )");
    uint32_t *d_quad_len=static_cast<uint32_t *>(db_quad_len->data());

    //scatter 0..num_org_pair to d_quad_offset using d_num_sums as map 
    thrust::scatter(exec_policy,thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0)+num_org_pair,d_num_sums,d_quad_offset);

if(0)
{
    std::cout<<"preprocess:d_quad_offset (after scatter)"<<std::endl;
    thrust::device_ptr<uint32_t> d_quad_offset_ptr=thrust::device_pointer_cast(d_quad_offset);
    thrust::copy(d_quad_offset_ptr,d_quad_offset_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}
    //copy idx of orginal pairs to all sub-pairs
    thrust::inclusive_scan(exec_policy,d_quad_offset,d_quad_offset+num_pq_pair,d_quad_offset,thrust::maximum<int>());

    //d_num_sums is no longer needed, delete db_num_sums and release its associated memory
    delete db_num_sums; db_num_sums=nullptr;

if(0)
{
    std::cout<<"preprocess: d_quad_offset (after scan )"<<std::endl;
    thrust::device_ptr<uint32_t> d_quad_offset_ptr=thrust::device_pointer_cast(d_quad_offset);
    thrust::copy(d_quad_offset_ptr,d_quad_offset_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

    //gather polygon idx and quadrant idx from original pairs into sub-pairs 
    thrust::gather(exec_policy,d_quad_offset,d_quad_offset+num_pq_pair,d_org_poly_idx,d_pq_poly_idx);
    thrust::gather(exec_policy,d_quad_offset,d_quad_offset+num_pq_pair,d_org_quad_idx,d_pq_quad_idx);

if(0)
{
    std::cout<<"preprocess: d_pq_poly_idx (after gather )"<<std::endl;

    thrust::device_ptr<uint32_t> d_poly_idx_ptr=thrust::device_pointer_cast(d_pq_poly_idx);
    thrust::copy(d_poly_idx_ptr,d_poly_idx_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"preprocess: d_pq_poly_idx (after gather )"<<std::endl;
    thrust::device_ptr<uint32_t> d_quad_idx_ptr=thrust::device_pointer_cast(d_pq_quad_idx);
    thrust::copy(d_quad_idx_ptr,d_quad_idx_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

    //allocate memory to store numbers of points in polygons in all sub-pairs and initialize them to 0 
    rmm::device_buffer *db_num_hits = new rmm::device_buffer(num_pq_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_num_hits!=nullptr, "Error allocating memory for array of numbers of points in polygons in all sub-pairs)");
    uint32_t *d_num_hits=static_cast<uint32_t *>(db_num_hits->data());
    HANDLE_CUDA_ERROR( cudaMemset(d_num_hits,0,num_pq_pair*sizeof(uint32_t)) );

    //generate offsets of sub-paris within the orginal pairs
    thrust::exclusive_scan_by_key(exec_policy,d_quad_offset,d_quad_offset+num_pq_pair,
        thrust::constant_iterator<int>(1),d_quad_offset);

    //assemble components in input/output iterators; note d_quad_offset used in both input and output
    auto qid_bid_iter=thrust::make_zip_iterator(thrust::make_tuple(d_pq_quad_idx,d_quad_offset));
    auto offset_length_iter=thrust::make_zip_iterator(thrust::make_tuple(d_quad_offset,d_quad_len));
    thrust::transform(exec_policy,qid_bid_iter,qid_bid_iter+num_pq_pair,
        offset_length_iter,gen_offset_length(threads_per_block,d_qt_length));

if(0)
{
    std::cout<<"preprocess: complete result"<<std::endl;
    std::cout<<"d_pq_poly_idx"<<std::endl;
    thrust::device_ptr<uint32_t> d_poly_idx_ptr=thrust::device_pointer_cast(d_pq_poly_idx);
    thrust::copy(d_poly_idx_ptr,d_poly_idx_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 

    std::cout<<"d_pq_quad_idx"<<std::endl;
    thrust::device_ptr<uint32_t> d_quad_idx_ptr=thrust::device_pointer_cast(d_pq_quad_idx);
    thrust::copy(d_quad_idx_ptr,d_quad_idx_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 

    std::cout<<"d_quad_offset"<<std::endl;
    thrust::device_ptr<uint32_t> d_quad_offset_ptr=thrust::device_pointer_cast(d_quad_offset);
    thrust::copy(d_quad_offset_ptr,d_quad_offset_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 

    std::cout<<"d_quad_length"<<std::endl;
    thrust::device_ptr<uint32_t> d_quad_len_ptr=thrust::device_pointer_cast(d_quad_len);
    thrust::copy(d_quad_len_ptr,d_quad_len_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 

}

     timeval t0,t1,t2,t3;
     gettimeofday(&t0, nullptr); 
     std::cout<<"running quad_pip_phase1_kernel"<<std::endl;
     quad_pip_phase1_kernel<T> <<< num_pq_pair, threads_per_block >>>(
        const_cast<uint32_t*>(d_pq_poly_idx),const_cast<uint32_t*>(d_pq_quad_idx),
        const_cast<uint32_t*>(d_quad_offset),const_cast<uint32_t*>(d_quad_len),
        d_qt_fpos,d_pnt_x,d_pnt_y,d_poly_fpos,d_poly_rpos,d_poly_x,d_poly_y,d_num_hits);
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    gettimeofday(&t1, nullptr); 
    float refine_phase1_time=cuspatial::calc_time("refine_phase1_time (ms) = ",t0,t1);

if(0)
{
    std::cout<<"phase1 results: d_num_hits (before reduce)"<<std::endl;
    thrust::device_ptr<uint32_t> d_num_hits_ptr=thrust::device_pointer_cast(d_num_hits);
    thrust::copy(d_num_hits_ptr,d_num_hits_ptr+num_pq_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}
 
    //remove poly-quad pair with zero hits
    auto valid_pq_pair_iter=thrust::make_zip_iterator(thrust::make_tuple(d_pq_poly_idx,d_pq_quad_idx,d_quad_offset,d_quad_len,d_num_hits));
    uint32_t num_valid_pair=thrust::remove_if(exec_policy,valid_pq_pair_iter,valid_pq_pair_iter+num_pq_pair,
    valid_pq_pair_iter,pq_remove_zero())-valid_pq_pair_iter;   
    std::cout<<"num_valid_pair="<<num_valid_pair<<std::endl;

if(0)
{
    std::cout<<"phase d_num_hits (after removal)"<<std::endl;
    thrust::device_ptr<uint32_t> d_num_hits_ptr=thrust::device_pointer_cast(d_num_hits);
    printf("d_num_hits: before reduce\n");
        thrust::copy(d_num_hits_ptr,d_num_hits_ptr+num_valid_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
 }
      
     uint32_t total_hits=thrust::reduce(exec_policy,d_num_hits,d_num_hits+num_valid_pair);
     std::cout<<"total_hits="<<total_hits<<std::endl;
     
     //prefix sum on numbers to generate offsets
     thrust::exclusive_scan(exec_policy,d_num_hits,d_num_hits+num_valid_pair,d_num_hits);

     gettimeofday(&t2, nullptr); 
     float refine_rebalance_time=cuspatial::calc_time("refine_rebalance_time(ms) = ",t1,t2);
 
if(0)
{
    std::cout<<"phase1 results:d_num_hits(after reduce)"<<std::endl;
    thrust::device_ptr<uint32_t> d_num_hits_ptr=thrust::device_pointer_cast(d_num_hits);
    thrust::copy(d_num_hits_ptr,d_num_hits_ptr+num_valid_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
}

    //use arrays in poly_idx and pnt_idx columns as kernel arguments to directly write output to columns
    std::unique_ptr<cudf::column> poly_idx_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), total_hits,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint32_t *d_res_poly_idx=cudf::mutable_column_device_view::create(poly_idx_col->mutable_view(), stream)->data<uint32_t>();
    CUDF_EXPECTS(d_res_poly_idx!=nullptr,"poly_idx can not be nullptr"); 

    std::unique_ptr<cudf::column> pnt_idx_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), total_hits,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint32_t *d_res_pnt_idx=cudf::mutable_column_device_view::create(pnt_idx_col->mutable_view(), stream)->data<uint32_t>();
    CUDF_EXPECTS(d_res_pnt_idx!=nullptr,"point_id can not be nullptr"); 

     std::cout<<"running quad_pip_phase2_kernel"<<std::endl;
     quad_pip_phase2_kernel<T> <<< num_valid_pair, threads_per_block >>>(
        d_pq_poly_idx,d_pq_quad_idx,
        d_quad_offset,d_quad_len,
        d_qt_fpos,d_pnt_x,d_pnt_y,
        d_poly_fpos,d_poly_rpos,d_poly_x,d_poly_y,
        d_num_hits,d_res_poly_idx,d_res_pnt_idx);
     HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
     gettimeofday(&t3, nullptr); 
     float refine_phase2_time=cuspatial::calc_time("refine_phase2_time(ms) = ",t2,t3);

if(0)
{
    std::cout<<"phase2 results:d_res_poly_id"<<std::endl;
    thrust::device_ptr<uint32_t> d_res_poly_ptr=thrust::device_pointer_cast(d_res_poly_idx);
    thrust::copy(d_res_poly_ptr,d_res_poly_ptr+total_hits,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"phase2 results:d_res_pnt_idx"<<std::endl;
    thrust::device_ptr<uint32_t> d_res_pnt_ptr=thrust::device_pointer_cast(d_res_pnt_idx);
    thrust::copy(d_res_pnt_ptr,d_res_pnt_ptr+total_hits,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

   std::vector<std::unique_ptr<cudf::column>> pair_cols;
   pair_cols.push_back(std::move(poly_idx_col));
   pair_cols.push_back(std::move(pnt_idx_col));
   return pair_cols;
}

struct pip_refine_processor {

    template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
    std::unique_ptr<cudf::experimental::table> operator()(
        cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
        cudf::column_view const& poly_fpos,cudf::column_view const& poly_rpos,
        cudf::column_view const& poly_x,cudf::column_view const& poly_y,
        rmm::mr::device_memory_resource* mr,cudaStream_t stream)
    {
        const uint32_t *d_poly_fpos=poly_fpos.data<uint32_t>();
        const uint32_t *d_poly_rpos=poly_rpos.data<uint32_t>();
        const T *d_poly_x=poly_x.data<T>();
        const T *d_poly_y=poly_y.data<T>();

        const T *d_pnt_x=pnt.column(0).data<T>();
        const T *d_pnt_y=pnt.column(1).data<T>();

        const uint32_t *d_qt_key=    quadtree.column(0).data<uint32_t>();
        const uint8_t  *d_qt_lev=    quadtree.column(1).data<uint8_t>();
        const bool     *d_qt_sign=   quadtree.column(2).data<bool>();
        const uint32_t *d_qt_length= quadtree.column(3).data<uint32_t>();
        const uint32_t *d_qt_fpos=   quadtree.column(4).data<uint32_t>();

        const uint32_t *d_pq_poly_id=   pq_pair.column(0).data<uint32_t>();
        const uint32_t *d_pq_quad_id=   pq_pair.column(1).data<uint32_t>();

        uint32_t num_pair=pq_pair.num_rows();
        uint32_t num_node=quadtree.num_rows();
        uint32_t num_poly=poly_fpos.size();
        uint32_t num_pnt=pnt.num_rows();

        std::vector<std::unique_ptr<cudf::column>> pair_cols= dowork(
            num_pair,d_pq_poly_id,d_pq_quad_id,
            num_node,d_qt_key,d_qt_lev,d_qt_sign,d_qt_length,d_qt_fpos,
            num_pnt,d_pnt_x,d_pnt_y,
            num_poly,d_poly_fpos,d_poly_rpos,d_poly_x,d_poly_y,
            mr,stream);

        std::unique_ptr<cudf::experimental::table> destination_table = 
            std::make_unique<cudf::experimental::table>(std::move(pair_cols));      

        return destination_table;
    }

    template<typename T, std::enable_if_t<!std::is_floating_point<T>::value >* = nullptr>
    std::unique_ptr<cudf::experimental::table> operator()(
        cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
        cudf::column_view const& poly_fpos,cudf::column_view const& poly_rpos,
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
    cudf::column_view const& poly_fpos,cudf::column_view const& poly_rpos,
    cudf::column_view const& poly_x,cudf::column_view const& poly_y)
{
    CUDF_EXPECTS(pq_pair.num_columns()==2,"a quadrant-polygon table must have 2 columns");
    CUDF_EXPECTS(quadtree.num_columns()==5,"a quadtree table must have 5 columns");
    CUDF_EXPECTS(pnt.num_columns()==2,"a point table must have 5 columns");
    CUDF_EXPECTS(poly_fpos.size()>0,"number of polygons must be greater than 0");
    CUDF_EXPECTS(poly_rpos.size()>=poly_fpos.size(),"number of rings must be no less than number of polygons");
    CUDF_EXPECTS(poly_x.size()==poly_y.size(),"numbers of vertices must be the same for both x and y columns");
    CUDF_EXPECTS(poly_x.size()>=4*poly_rpos.size(),"all rings must have at least 4 vertices");

    cudf::data_type pnt_dtype=pnt.column(0).type();
    cudf::data_type poly_dtype=poly_x.type();
    CUDF_EXPECTS(pnt_dtype==poly_dtype,"point and polygon must have the same data type");

    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    return cudf::experimental::type_dispatcher(pnt_dtype,pip_refine_processor{},
    pq_pair,quadtree,pnt,poly_fpos,poly_rpos,poly_x,poly_y,mr,stream);
}

}// namespace cuspatial
