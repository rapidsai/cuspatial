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

typedef thrust::tuple<double, double,double,double,double,uint32_t,uint32_t> quad_point_parameters;

/*

*/
template<typename T>
std::vector<std::unique_ptr<cudf::column>> dowork(
    uint32_t num_node,const uint32_t *d_p_qtkey,const uint8_t *d_p_qtlev,
    const bool *d_p_qtsign, const uint32_t *d_p_qtlength, const uint32_t *d_p_qtfpos,
    const uint32_t num_poly,const T *poly_x1, const T *poly_y1,T const *poly_x2, const T *poly_y2,
    const SBBox<double>& aoi_bbox, double scale,uint32_t num_level, uint32_t min_size, 
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)    
                                         
{
    double x1=thrust::get<0>(aoi_bbox.first);
    double y1=thrust::get<1>(aoi_bbox.first);
    double x2=thrust::get<0>(aoi_bbox.second);
    double y2=thrust::get<1>(aoi_bbox.second);

    std::cout<<"num_node="<<num_node<<std::endl;
    std::cout<<"num_poly="<<num_poly<<std::endl;
    std::cout<<"bounding box(x1,y1,x2,y2)=("<<x1<<","<<y1<<","<<x2<<","<<x2<<","<<y2<<std::endl;
    std::cout<<"scale="<<scale<<std::endl;
    std::cout<<"num_level="<<num_level<<std::endl;
    std::cout<<"match: min_size="<<min_size<<std::endl;

    auto exec_policy = rmm::exec_policy(stream)->on(stream);

    SBBox<T> * d_poly_sbbox=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_poly_sbbox),num_poly*sizeof(SBBox<T>), 0));
    CUDF_EXPECTS(d_poly_sbbox!=NULL,"Error allocating memory for polygon bounding boxes on device");
 
 if(0)
 {
    std::cout<<"x1"<<std::endl;
    thrust::device_ptr<const T> d_x1_ptr = thrust::device_pointer_cast(poly_x1);
    thrust::copy(d_x1_ptr,d_x1_ptr+num_poly,std::ostream_iterator<T>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"y1"<<std::endl;
    thrust::device_ptr<const T> d_y1_ptr = thrust::device_pointer_cast(poly_y1);
    thrust::copy(d_y1_ptr,d_y1_ptr+num_poly,std::ostream_iterator<T>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"x2"<<std::endl;
    thrust::device_ptr<const T> d_x2_ptr = thrust::device_pointer_cast(poly_x2);
    thrust::copy(d_x2_ptr,d_x2_ptr+num_poly,std::ostream_iterator<T>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"y2"<<std::endl;
    thrust::device_ptr<const T> d_y2_ptr = thrust::device_pointer_cast(poly_y2);
    thrust::copy(d_y2_ptr,d_y2_ptr+num_poly,std::ostream_iterator<T>(std::cout, " "));std::cout<<std::endl;
}

if(0)
{
   std::cout<<"qt sign"<<std::endl;
   thrust::device_ptr<const bool> d_sign_ptr=thrust::device_pointer_cast(d_p_qtsign);
   thrust::copy(d_sign_ptr,d_sign_ptr+num_node,std::ostream_iterator<bool>(std::cout, " "));std::cout<<std::endl;
}

    auto ploy_bbox_iter=thrust::make_zip_iterator(
        thrust::make_tuple(poly_x1,poly_y1,poly_x2,poly_y2));
    thrust::transform(exec_policy,ploy_bbox_iter,ploy_bbox_iter+num_poly,d_poly_sbbox,tuple2bbox<T>());

    uint32_t num_top_lev_children=thrust::count_if(exec_policy,d_p_qtlev,
        d_p_qtlev+num_node,thrust::placeholders::_1==0);  
    uint32_t init_len=1000;
    uint32_t curr_cap=init_len;
    
    uint8_t *d_pq_lev_out=NULL,*d_pq_type_out=NULL;
    RMM_TRY( RMM_ALLOC( &d_pq_lev_out,curr_cap* sizeof(uint8_t), stream));
    CUDF_EXPECTS(d_pq_lev_out!=NULL,"Error allocating memory for permanent level array on device");
    RMM_TRY( RMM_ALLOC( &d_pq_type_out,curr_cap* sizeof(uint8_t), stream));
    CUDF_EXPECTS(d_pq_type_out!=NULL,"Error allocating memory for permanent type array on device");

    uint32_t *d_quad_idx_out=NULL,*d_poly_idx_out=NULL;
    RMM_TRY( RMM_ALLOC( &d_quad_idx_out,curr_cap* sizeof(uint32_t), stream));
    CUDF_EXPECTS(d_quad_idx_out!=NULL,"Error allocating memory for permanent quadrant index array on device");
    RMM_TRY( RMM_ALLOC( &d_poly_idx_out,curr_cap* sizeof(uint32_t), stream));
    CUDF_EXPECTS(d_poly_idx_out!=NULL,"Error allocating memory for permanent polygon index array on device");

    auto pair_output_iter=thrust::make_zip_iterator(
        thrust::make_tuple(d_pq_lev_out,d_pq_type_out,d_poly_idx_out,d_quad_idx_out));
    uint32_t output_nodes_pos=0;

    uint32_t  num_pair=num_top_lev_children*num_poly;  
    std::cout<<"num_top_lev_children="<<num_top_lev_children<<std::endl;
    std::cout<<"num_poly="<<num_poly<<std::endl;
    std::cout<<"num_pair="<<num_pair<<std::endl;

    uint8_t *d_pq_lev_temp=NULL,*d_pq_type_temp=NULL;
    RMM_TRY( RMM_ALLOC( &d_pq_lev_temp,num_pair* sizeof(uint8_t), stream));
    CUDF_EXPECTS(d_pq_lev_temp!=NULL,"Error allocating memory for temporal level array on device");
    RMM_TRY( RMM_ALLOC( &d_pq_type_temp,num_pair* sizeof(uint8_t), stream));
    CUDF_EXPECTS(d_pq_type_temp!=NULL,"Error allocating memory for temporal type array on device");

    uint32_t *d_quad_idx_temp=NULL,*d_poly_idx_temp=NULL;
    RMM_TRY( RMM_ALLOC( &d_quad_idx_temp,num_pair* sizeof(uint32_t), stream));
    CUDF_EXPECTS(d_quad_idx_temp!=NULL,"Error allocating memory for temporal quadrant index array on device");
    RMM_TRY( RMM_ALLOC( &d_poly_idx_temp,num_pair* sizeof(uint32_t), stream));
    CUDF_EXPECTS(d_poly_idx_temp!=NULL,"Error allocating memory for temporal polygon index array on device");
 
    auto pair_counting_iter=thrust::make_counting_iterator(0);
    auto pair_output_temp_iter=thrust::make_zip_iterator(
        thrust::make_tuple(d_pq_lev_temp,d_pq_type_temp,d_poly_idx_temp,d_quad_idx_temp));

    thrust::transform(exec_policy,pair_counting_iter,pair_counting_iter+num_pair,pair_output_temp_iter,
        pairwise_test_intersection<T>(num_level,num_top_lev_children,aoi_bbox,scale,d_p_qtkey,d_p_qtlev,d_p_qtsign,d_poly_sbbox));

    uint32_t num_leaf_pair=thrust::copy_if(exec_policy,pair_output_temp_iter,pair_output_temp_iter+num_pair,
        pair_output_iter+output_nodes_pos,qt_is_type(0))-(pair_output_iter+output_nodes_pos);

    uint32_t num_nonleaf_pair=thrust::remove_if(exec_policy,pair_output_temp_iter,pair_output_temp_iter+num_pair,
        pair_output_temp_iter,qt_not_type(1))-pair_output_temp_iter;
    std::cout<<"num_leaf_pair="<<num_leaf_pair<<" ,num_nonleaf_pair="<<num_nonleaf_pair<<std::endl;

    output_nodes_pos+=num_leaf_pair; 
    for(uint32_t i=1;i<num_level;i++)
    {
        uint32_t *d_quad_nchild=NULL;
        RMM_TRY( RMM_ALLOC( &d_quad_nchild,num_nonleaf_pair* sizeof(uint32_t), stream));
        CUDF_EXPECTS(d_quad_nchild!=NULL,"Error allocating memory for number of child nodes array on device");

        thrust::transform(exec_policy,d_quad_idx_temp,d_quad_idx_temp+num_nonleaf_pair,
              d_quad_nchild,get_vec_element<const uint32_t>(d_p_qtlength));

        num_pair=thrust::reduce(exec_policy,d_quad_nchild,d_quad_nchild+num_nonleaf_pair);
        std::cout<<"num_pair after gathering child nodes="<<num_pair<<std::endl;

        uint32_t *d_expand_pos=NULL;
        RMM_TRY( RMM_ALLOC( &d_expand_pos,num_pair* sizeof(uint32_t), stream));
        CUDF_EXPECTS(d_expand_pos!=NULL,"Error allocating memory for expanded offset array on device");
        HANDLE_CUDA_ERROR( cudaMemset(d_expand_pos,0,num_pair) );

        uint8_t *d_pq_lev_new=NULL,*d_pq_type_new=NULL;
        RMM_TRY( RMM_ALLOC( &d_pq_lev_new,num_pair* sizeof(uint8_t), stream));
        CUDF_EXPECTS(d_pq_lev_new!=NULL,"Error allocating memory for expanded lev array on device");
        RMM_TRY( RMM_ALLOC( &d_pq_type_new,num_pair* sizeof(uint8_t),stream));
        CUDF_EXPECTS(d_pq_type_new!=NULL,"Error allocating memory for expanded type array on device");
  
        uint32_t *d_quad_idx_new=NULL,*d_poly_idx_new=NULL;
        RMM_TRY( RMM_ALLOC( &d_quad_idx_new,num_pair* sizeof(uint32_t), stream));
        CUDF_EXPECTS(d_quad_idx_new!=NULL,"Error allocating memory for expanded quadrant index  array on device");      
        RMM_TRY( RMM_ALLOC( &d_poly_idx_new,num_pair* sizeof(uint32_t), stream));
        CUDF_EXPECTS(d_poly_idx_new!=NULL,"Error allocating memory for expanded polygon index  array on device");

        auto pair_output_new_iter=thrust::make_zip_iterator(thrust::make_tuple(d_pq_lev_new,d_pq_type_new,d_poly_idx_new,d_quad_idx_new));
        auto counting_iter=thrust::make_counting_iterator(0);
        thrust::exclusive_scan(exec_policy,d_quad_nchild,d_quad_nchild+num_nonleaf_pair,d_quad_nchild);
        thrust::scatter(exec_policy,counting_iter,counting_iter+num_nonleaf_pair,d_quad_nchild,d_expand_pos);
        RMM_TRY(RMM_FREE(d_quad_nchild,stream));d_quad_nchild=NULL;

        thrust::inclusive_scan(exec_policy,d_expand_pos,d_expand_pos+num_pair,d_expand_pos,thrust::maximum<int>());
        thrust::gather(exec_policy,d_expand_pos,d_expand_pos+num_pair,pair_output_temp_iter,pair_output_new_iter);
        
        uint32_t *d_seq_pos=NULL;
        RMM_TRY( RMM_ALLOC( &d_seq_pos,num_pair* sizeof(uint32_t), stream));
        CUDF_EXPECTS(d_seq_pos!=NULL,"Error allocating memory for sequence index  array on device");

        thrust::exclusive_scan_by_key(exec_policy,d_expand_pos,d_expand_pos+num_pair,
            thrust::constant_iterator<int>(1),d_seq_pos);

        RMM_TRY(RMM_FREE(d_expand_pos,stream));d_expand_pos=NULL;
        auto update_quad_iter=thrust::make_zip_iterator(thrust::make_tuple(d_quad_idx_new,thrust::make_counting_iterator(0)));
        thrust::transform(exec_policy,update_quad_iter,update_quad_iter+num_pair,d_quad_idx_new,
            update_quad(d_p_qtfpos,d_seq_pos));

        RMM_TRY(RMM_FREE(d_seq_pos,stream));d_seq_pos=NULL;

        auto pq_pair_iterator=thrust::make_zip_iterator(thrust::make_tuple(d_poly_idx_new,d_quad_idx_new));
        thrust::transform(exec_policy,pq_pair_iterator,pq_pair_iterator+num_pair,pair_output_new_iter,
            twolist_test_intersection<T>(num_level,aoi_bbox,scale,d_p_qtkey,d_p_qtlev,d_p_qtsign,d_poly_sbbox));

        num_leaf_pair=thrust::copy_if(exec_policy,pair_output_new_iter,pair_output_new_iter+num_pair,
            pair_output_iter+output_nodes_pos,qt_is_type(0))-(pair_output_iter+output_nodes_pos);
        num_nonleaf_pair=thrust::remove_if(exec_policy,pair_output_new_iter,pair_output_new_iter+num_pair,
            pair_output_new_iter,qt_not_type(1))-pair_output_new_iter;

        std::cout<<"level="<<i<<std::endl;
        std::cout<<"num_leaf_pair="<<num_leaf_pair<<std::endl;
        std::cout<<"num_nonleaf_pair="<<num_nonleaf_pair<<std::endl;

        output_nodes_pos+=num_leaf_pair;

        RMM_TRY( RMM_FREE(d_pq_lev_temp,stream) ); d_pq_lev_temp=d_pq_lev_new;
        RMM_TRY( RMM_FREE(d_pq_type_temp,stream) ); d_pq_type_temp=d_pq_type_new;
        RMM_TRY( RMM_FREE(d_poly_idx_temp,stream) ); d_poly_idx_temp=d_poly_idx_new;
        RMM_TRY( RMM_FREE(d_quad_idx_temp,stream) ); d_quad_idx_temp=d_quad_idx_new;

        if(num_nonleaf_pair==0) 
            break;

        pair_output_temp_iter=thrust::make_zip_iterator(thrust::make_tuple(d_pq_lev_temp,d_pq_type_temp,d_poly_idx_temp,d_quad_idx_temp));

        //expand child nodes and get ready for the next round
        uint32_t max_num=output_nodes_pos+num_nonleaf_pair*4;
        if((i<num_level-1)&&(max_num>curr_cap))
        {
            curr_cap*=((max_num/curr_cap)+1);
            std::cout<<"increasing output capacity: level="<<i<<" to "<<curr_cap<<std::endl;

            uint8_t *d_pq_lev_new=NULL,*d_pq_type_new=NULL;
            RMM_TRY( RMM_ALLOC( &d_pq_lev_new,curr_cap* sizeof(uint8_t), stream));
            CUDF_EXPECTS(d_pq_lev_new!=NULL,"Error allocating memory for increased lev array on device");
            HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_pq_lev_new, (void *)d_pq_lev_out, output_nodes_pos * sizeof(uint8_t), cudaMemcpyDeviceToDevice ) );
            RMM_TRY( RMM_FREE (d_pq_lev_out,stream));      
            d_pq_lev_out=d_pq_lev_new;

            RMM_TRY( RMM_ALLOC( &d_pq_type_new,curr_cap* sizeof(uint8_t), stream));
            CUDF_EXPECTS(d_pq_type_new!=NULL,"Error allocating memory for increased type array on device");
            HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_pq_type_new, (void *)d_pq_type_out, output_nodes_pos * sizeof(uint8_t), cudaMemcpyDeviceToDevice ) );
            RMM_TRY( RMM_FREE (d_pq_type_out,stream));
            d_pq_type_out=d_pq_type_new;

            uint32_t *d_quad_idx_new=NULL,*d_poly_idx_new=NULL;
            RMM_TRY( RMM_ALLOC( &d_quad_idx_new,curr_cap* sizeof(uint32_t), stream));
            CUDF_EXPECTS(d_quad_idx_new!=NULL,"Error allocating memory for increased quad index array on device");
            HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_quad_idx_new, (void *)d_quad_idx_out, output_nodes_pos * sizeof(uint32_t), cudaMemcpyDeviceToDevice ) );
            RMM_TRY( RMM_FREE (d_quad_idx_out,stream));
            d_quad_idx_out=d_quad_idx_new;

            RMM_TRY( RMM_ALLOC( &d_poly_idx_new,curr_cap* sizeof(uint32_t), stream));
            CUDF_EXPECTS(d_poly_idx_new!=NULL,"Error allocating memory for increased polygon index array on device");
            HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_poly_idx_new, (void *)d_poly_idx_out, output_nodes_pos * sizeof(uint32_t), cudaMemcpyDeviceToDevice ) );
            RMM_TRY( RMM_FREE (d_poly_idx_out,stream));
            d_poly_idx_out=d_poly_idx_new;

            pair_output_iter=thrust::make_zip_iterator(thrust::make_tuple(d_pq_lev_out,d_pq_type_out,d_poly_idx_out,d_quad_idx_out));
        }
        std::cout<<"level="<<i<<" output_nodes_pos="<<output_nodes_pos<<" curr_cap="<<curr_cap<<std::endl;
    }
    std::cout<<"final: output_nodes_pos="<<output_nodes_pos;
    CUDF_EXPECTS(output_nodes_pos<=curr_cap,"output arrays: out of boundary"); 
    RMM_TRY(RMM_FREE(d_poly_sbbox,stream));d_poly_sbbox=NULL;    

    std::unique_ptr<cudf::column> poly_idx_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), output_nodes_pos,cudf::mask_state::UNALLOCATED,  stream, mr);
    uint32_t *d_pq_poly_idx=cudf::mutable_column_device_view::create(poly_idx_col->mutable_view(), stream)->data<uint32_t>();
    CUDF_EXPECTS(d_pq_poly_idx!=NULL,"Error in accessing data array of polygon index column"); 
    thrust::copy(exec_policy,d_poly_idx_out,d_poly_idx_out+output_nodes_pos,d_pq_poly_idx);

    std::unique_ptr<cudf::column> quad_idx_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), output_nodes_pos,cudf::mask_state::UNALLOCATED,  stream, mr);
    uint32_t *d_pq_quad_idx=cudf::mutable_column_device_view::create(quad_idx_col->mutable_view(), stream)->data<uint32_t>();
    CUDF_EXPECTS(d_pq_quad_idx!=NULL,"Error in accessing data array of quadrant index column"); 
    thrust::copy(exec_policy,d_quad_idx_out,d_quad_idx_out+output_nodes_pos,d_pq_quad_idx);

    RMM_FREE(d_pq_lev_out,stream);d_pq_lev_out=NULL;
    RMM_FREE(d_pq_type_out,stream);d_pq_lev_out=NULL;
    RMM_FREE(d_poly_idx_out,stream);d_poly_idx_out=NULL;
    RMM_FREE(d_quad_idx_out,stream);d_quad_idx_out=NULL;

if(0)
{
    std::cout<<"total pairs="<<output_nodes_pos<<std::endl;

    thrust::device_ptr<uint32_t> d_poly_idx_ptr=thrust::device_pointer_cast(d_pq_poly_idx);
    std::cout<<"poly id of poly-quad pairs"<<std::endl;
    thrust::copy(d_poly_idx_ptr,d_poly_idx_ptr+output_nodes_pos,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

    thrust::device_ptr<uint32_t> d_quad_idx_ptr=thrust::device_pointer_cast(d_pq_quad_idx);
    std::cout<<"quadrant id of poly-quad pairs"<<std::endl;
    thrust::copy(d_quad_idx_ptr,d_quad_idx_ptr+output_nodes_pos,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

    std::vector<std::unique_ptr<cudf::column>> pair_cols;
    pair_cols.push_back(std::move(poly_idx_col));
    pair_cols.push_back(std::move(quad_idx_col));
    return pair_cols;
}

struct quad_bbox_processor {
    template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
    std::unique_ptr<cudf::experimental::table> operator()(
       const cudf::table_view& quadtree,const cudf::table_view& poly_bbox,    
       quad_point_parameters qpi,
       rmm::mr::device_memory_resource* mr,
        cudaStream_t stream)
    {
        double x1=thrust::get<0>(qpi);
        double y1=thrust::get<1>(qpi);
        double x2=thrust::get<2>(qpi);
        double y2=thrust::get<3>(qpi);
        SBBox<double> aoi_bbox(thrust::make_tuple(x1,y1),thrust::make_tuple(x2,y2));
        std::cout<<"quadtree_poly.aoi:"<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<std::endl;
        double scale=thrust::get<4>(qpi);
        uint32_t num_level=thrust::get<5>(qpi);
        uint32_t min_size=thrust::get<6>(qpi);    

        const uint32_t *d_p_qtkey=    quadtree.column(0).data<uint32_t>();
        const uint8_t  *d_p_qtlev=    quadtree.column(1).data<uint8_t>();
        const bool     *d_p_qtsign=   quadtree.column(2).data<bool>();
        const uint32_t *d_p_qtlength= quadtree.column(3).data<uint32_t>();
        const uint32_t *d_p_qtfpos=   quadtree.column(4).data<uint32_t>();

        const T *poly_x1=poly_bbox.column(0).data<T>();
        const T *poly_y1=poly_bbox.column(1).data<T>();
        const T *poly_x2=poly_bbox.column(2).data<T>();
        const T *poly_y2=poly_bbox.column(3).data<T>();     

        uint32_t num_node=quadtree.num_rows();
        uint32_t num_poly=poly_bbox.num_rows();
        std::vector<std::unique_ptr<cudf::column>> pair_cols=dowork(
            num_node,d_p_qtkey,d_p_qtlev,d_p_qtsign,d_p_qtlength,d_p_qtfpos,
            num_poly,poly_x1,poly_y1,poly_x2,poly_y2,
            aoi_bbox,scale,num_level,min_size,mr,stream);

      std::unique_ptr<cudf::experimental::table> destination_table = 
        std::make_unique<cudf::experimental::table>(std::move(pair_cols));      

      return destination_table;
    }

  template<typename T, std::enable_if_t<!std::is_floating_point<T>::value >* = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(
    const cudf::table_view& quadtree,const cudf::table_view& bbox,    
    quad_point_parameters qpi,
    rmm::mr::device_memory_resource* mr,
        cudaStream_t stream)
    {
     CUDF_FAIL("Non-floating point operation is not supported");
    }  
      
};
  
} //end anonymous namespace

namespace cuspatial {

std::unique_ptr<cudf::experimental::table> quad_bbox_join(
    cudf::table_view const& quadtree,cudf::table_view const& poly_bbox,
    double x1,double y1,double x2,double y2, double scale, uint32_t num_level, uint32_t min_size)
{   
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    quad_point_parameters qpi=thrust::make_tuple(x1,y1,x2,y2,scale,num_level,min_size);

    cudf::data_type dtype=poly_bbox.column(0).type();

    return cudf::experimental::type_dispatcher(dtype,quad_bbox_processor{}, 
        quadtree,poly_bbox,qpi, mr,stream);
}

}// namespace cuspatial
