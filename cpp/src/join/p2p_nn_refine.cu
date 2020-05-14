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
__global__ void kernel_nearest_polyline(
        const uint32_t *quad_idxs, //point quadrant id array -base 0
        const uint32_t *pid_fpos, // starting positions of the first polyline idx
        const uint32_t *poly_idxs,

  	const uint32_t *qt_len, //numbers of points quadrants
        const uint32_t *qt_fpos, //positions of first points in quadrants
        const T *pnt_x,         
	const T *pnt_y,
       
        const uint32_t *poly_spos, //positions of the first vertex in a polyline
        const T *poly_x,
        const T *poly_y,
        
        uint32_t * out_pnt_id,
        uint32_t * out_poly_id,
        T * out_distance
        )
{
        //each block processes a quadrant
        uint32_t block_idx = blockIdx.x+gridDim.x*blockIdx.y;
        uint32_t quad_idx = quad_idxs[block_idx];
        
        uint32_t p_f = (block_idx==0)?0:pid_fpos[block_idx-1];
        uint32_t p_t = pid_fpos[block_idx];

if(threadIdx.x==0)
    printf("block_idx=%d quad_idx=%d p_f=%d p_t=%d\n",block_idx,quad_idx,p_f,p_t);
        
        uint32_t sz_points=(quad_idx==0)?0:qt_len[quad_idx-1];
        uint32_t base = 0;
        for (; base < sz_points; base+=blockDim.x) 
        {
            //each thread loads its point
            if (base + threadIdx.x < qt_len[quad_idx])
            {
                uint32_t p=qt_fpos[quad_idx]+base+threadIdx.x;
                T x = pnt_x[p];
                T y = pnt_y[p];
                T dist = 1e20;
                uint32_t nearest_id = (uint32_t)-1;
                
                for (uint32_t j = p_f; j < p_t; j++) //for each polyline
                {
                    uint32_t poly_idx = poly_idxs[j];
                    uint32_t v_f = (0 == poly_idx) ? 0 :poly_spos[poly_idx-1];
                    uint32_t v_t=poly_spos[poly_idx];

                    for (uint k = v_f; k < v_t-1; k++) //for each line
                    {
                        T x0 = poly_x[k]; 
                        T y0 = poly_y[k];
                        T x1 = poly_x[k+1];
                        T y1 = poly_y[k+1];
                        T dx = x1 - x0;
                        T dy = y1 - y0;
                        T dx2 = x - x0;
                        T dy2 = y - y0;
                        T r = (dx*dx2+dy*dy2)/sqrt(dx*dx+dy*dy);
                        T d = 1e20;
                        if (r <= 0 || r >= sqrt(dx*dx+dy*dy))
                        {
                            T d1 = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
                            T d2 = sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1));
                            d = (d < d1) ? d : d1;
                            d = (d < d2) ? d : d2;
                        }
                        else
                        {
                            d = sqrt((dx2*dx2+dy2*dy2)-(r*r));
                        }
                        if (d < dist)
                        {
                            dist = d;
                            nearest_id = poly_idx;
                        }
                    }
                }
                //__syncthreads();
                //TODO: use input point id
                out_pnt_id[p]=p; 
                out_poly_id[p]=nearest_id;
                out_distance[p]=dist;
            }
        }
}

template<typename T>
std::vector<std::unique_ptr<cudf::column>> dowork(
    uint32_t num_pair,const uint32_t * d_poly_idx,const uint32_t * d_quad_idx,
    uint32_t num_node,const uint32_t *d_qt_key,const uint8_t *d_qt_lev,
    const bool *d_qt_sign, const uint32_t *d_qt_length, const uint32_t *d_qt_fpos,
    const uint32_t num_pnt,const T *d_pnt_x, const T *d_pnt_y,
    const uint32_t num_poly, const uint32_t * d_poly_spos,const T *d_poly_x, const T *d_poly_y,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)                                         
{
     auto exec_policy = rmm::exec_policy(stream);
     
    //sort (d_poly_idx,d_quad_idx) using d_quad_idx as key ==>(quad_idxs, poly_idxs)
    rmm::device_buffer *db_temp_poly_idx = new rmm::device_buffer(num_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_temp_poly_idx!=nullptr, "Error allocating memory for temporal poly_idx");
    uint32_t *d_temp_poly_idx=static_cast<uint32_t *>(db_temp_poly_idx->data());
    thrust::copy(exec_policy->on(stream),d_poly_idx,d_poly_idx+num_pair,d_temp_poly_idx);
 
    rmm::device_buffer *db_temp_quad_idx = new rmm::device_buffer(num_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_temp_quad_idx!=nullptr, "Error allocating memory for temporal quad_idx");
    uint32_t *d_temp_quad_idx=static_cast<uint32_t *>(db_temp_quad_idx->data());
    thrust::copy(exec_policy->on(stream),d_quad_idx,d_quad_idx+num_pair,d_temp_quad_idx);
 
    thrust::sort_by_key(exec_policy->on(stream),d_temp_quad_idx,d_temp_quad_idx+num_pair,d_temp_poly_idx);
if(1)
{
    std::cout<<"temp_quad_idx"<<std::endl;
    thrust::device_ptr<uint32_t> temp_quad_idx_ptr=thrust::device_pointer_cast(d_temp_quad_idx);
    thrust::copy(temp_quad_idx_ptr,temp_quad_idx_ptr+num_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"temp_poly_idx"<<std::endl;
    thrust::device_ptr<uint32_t> temp_poly_idx_ptr=thrust::device_pointer_cast(d_temp_poly_idx);
    thrust::copy(temp_poly_idx_ptr,temp_poly_idx_ptr+num_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
 }

    //reduce_by_key using d_quad_idx as the key 
    //exclusive_scan on numbers of polys associated with a quadrant==>pid_fpos 
  
    rmm::device_buffer *db_temp_pidx_fpos = new rmm::device_buffer(num_pair* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_temp_pidx_fpos!=nullptr, "Error allocating memory for pid_fpos");
    uint32_t *d_temp_pidx_fpos=static_cast<uint32_t *>(db_temp_pidx_fpos->data());
    
    uint32_t num_quads=thrust::reduce_by_key(exec_policy->on(stream),d_temp_quad_idx,d_temp_quad_idx+num_pair,
            thrust::constant_iterator<uint32_t>(1),d_temp_quad_idx, d_temp_pidx_fpos).second-d_temp_pidx_fpos;
    std::cout<<"num_quads="<<num_quads<<std::endl;
    thrust::inclusive_scan(exec_policy->on(stream),d_temp_pidx_fpos,d_temp_pidx_fpos+num_quads,d_temp_pidx_fpos);

if(1)
{
    std::cout<<"temp_quad_idx"<<std::endl;
    thrust::device_ptr<uint32_t> temp_quad_idx_ptr=thrust::device_pointer_cast(d_temp_quad_idx);
    thrust::copy(temp_quad_idx_ptr,temp_quad_idx_ptr+num_quads,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"temp_poly_idx"<<std::endl;
    thrust::device_ptr<uint32_t> temp_poly_idx_ptr=thrust::device_pointer_cast(d_temp_poly_idx);
    thrust::copy(temp_poly_idx_ptr,temp_poly_idx_ptr+num_pair,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"temp_pid_fpos"<<std::endl;
    thrust::device_ptr<uint32_t> temp_pid_fpos_ptr=thrust::device_pointer_cast(d_temp_pidx_fpos);
    thrust::copy(temp_pid_fpos_ptr,temp_pid_fpos_ptr+num_quads,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

    std::unique_ptr<cudf::column> pnt_idx_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), num_pnt,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint32_t *d_res_pnt_idx=cudf::mutable_column_device_view::create(pnt_idx_col->mutable_view(), stream)->data<uint32_t>();
    CUDF_EXPECTS(d_res_pnt_idx!=nullptr,"point_id can not be nullptr"); 

    std::unique_ptr<cudf::column> poly_idx_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), num_pnt,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint32_t *d_res_poly_idx=cudf::mutable_column_device_view::create(poly_idx_col->mutable_view(), stream)->data<uint32_t>();
    CUDF_EXPECTS(d_res_poly_idx!=nullptr,"poly_idx can not be nullptr"); 

    std::unique_ptr<cudf::column> poly_dist_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::FLOAT64), num_pnt,cudf::mask_state::UNALLOCATED,  stream, mr);      
    T *d_res_poly_dist=cudf::mutable_column_device_view::create(poly_dist_col->mutable_view(), stream)->data<T>();
    CUDF_EXPECTS(d_res_poly_dist!=nullptr,"poly_dist can not be nullptr"); 
  
    timeval t0,t1;
    gettimeofday(&t0, nullptr); 
    std::cout<<"running quad_pip_phase1_kernel"<<std::endl;
    kernel_nearest_polyline<T> <<< num_quads, threads_per_block >>>
    (
        const_cast<uint32_t*>(d_temp_quad_idx),
        const_cast<uint32_t*>(d_temp_pidx_fpos),
        const_cast<uint32_t*>(d_temp_poly_idx),
 
        const_cast<uint32_t*>(d_qt_length),
        const_cast<uint32_t*>(d_qt_fpos),
        const_cast<T *>(d_pnt_x),
        const_cast<T *>(d_pnt_y),     
        
        const_cast<uint32_t*>(d_poly_spos),
        const_cast<T *>(d_poly_x),
        const_cast<T *>(d_poly_y),
        
        d_res_pnt_idx,
        d_res_poly_idx,
        d_res_poly_dist
    );
    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );
    gettimeofday(&t1, nullptr); 
    float refine_phase1_time=cuspatial::calc_time("refine_phase1_time (ms) = ",t0,t1);
 
    delete db_temp_poly_idx;
    delete db_temp_quad_idx;
    delete db_temp_pidx_fpos;
 
    std::vector<std::unique_ptr<cudf::column>> res_cols;
    res_cols.push_back(std::move(pnt_idx_col));
    res_cols.push_back(std::move(poly_idx_col));
    res_cols.push_back(std::move(poly_dist_col));
    return res_cols;
}

struct nn_distance_processor {

    template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
    std::unique_ptr<cudf::experimental::table> operator()(
        cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
        cudf::column_view const& poly_spos,
        cudf::column_view const& poly_x,cudf::column_view const& poly_y,
        rmm::mr::device_memory_resource* mr,cudaStream_t stream)
    {
        const uint32_t *d_poly_spos=poly_spos.data<uint32_t>();
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
        uint32_t num_poly=poly_spos.size();
        uint32_t num_pnt=pnt.num_rows();
        
        std::cout<<"num_pair="<<num_pair<<" num_node="<<num_node<<" num_poly="<<num_poly<<" num_pnt"<<num_pnt<<std::endl;

        std::vector<std::unique_ptr<cudf::column>> res_cols= dowork(
            num_pair,d_pq_poly_id,d_pq_quad_id,
            num_node,d_qt_key,d_qt_lev,d_qt_sign,d_qt_length,d_qt_fpos,
            num_pnt,d_pnt_x,d_pnt_y,
            num_poly,d_poly_spos,d_poly_x,d_poly_y,
            mr,stream);

        std::unique_ptr<cudf::experimental::table> destination_table = 
            std::make_unique<cudf::experimental::table>(std::move(res_cols));      

        return destination_table;
    }

    template<typename T, std::enable_if_t<!std::is_floating_point<T>::value >* = nullptr>
    std::unique_ptr<cudf::experimental::table> operator()(
        cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
        cudf::column_view const& poly_spos,
        cudf::column_view const& poly_x,cudf::column_view const& poly_y,
        rmm::mr::device_memory_resource* mr, cudaStream_t stream)       
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }

};

} //end anonymous namespace

namespace cuspatial {
std::unique_ptr<cudf::experimental::table> p2p_nn_refine(
    cudf::table_view const& pq_pair,cudf::table_view const& quadtree,cudf::table_view const& pnt,
    cudf::column_view const& poly_spos,
    cudf::column_view const& poly_x,cudf::column_view const& poly_y)
{
    CUDF_EXPECTS(pq_pair.num_columns()==2,"a quadrant-polygon table must have 2 columns");
    CUDF_EXPECTS(quadtree.num_columns()==5,"a quadtree table must have 5 columns");
    CUDF_EXPECTS(pnt.num_columns()==2,"a point table must have 2 columns");
    CUDF_EXPECTS(poly_spos.size()>0,"number of polylines must be greater than 0");
    CUDF_EXPECTS(poly_x.size()==poly_y.size(),"numbers of vertices must be the same for both x and y columns");
    CUDF_EXPECTS(poly_x.size()>=2*poly_spos.size(),"all polylines must have at least two vertices");

    cudf::data_type pnt_dtype=pnt.column(0).type();
    cudf::data_type poly_dtype=poly_x.type();
    CUDF_EXPECTS(pnt_dtype==poly_dtype,"point and polygon must have the same data type");

    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    return cudf::experimental::type_dispatcher(pnt_dtype,nn_distance_processor{},
        pq_pair,quadtree,pnt,poly_spos,poly_x,poly_y,mr,stream);
}

}// namespace cuspatial
