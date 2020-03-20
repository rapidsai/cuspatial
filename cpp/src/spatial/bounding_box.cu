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
#include <utility/bbox_thrust.cuh>
#include <utility/bbox_thrust.cuh>
#include <cuspatial/bounding_box.hpp>

namespace
{

struct bounding_box_processor {

    template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
    std::unique_ptr<cudf::experimental::table> operator()(
        const cudf::column_view& fpos,const cudf::column_view& rpos,
        const cudf::column_view& x,const cudf::column_view& y,
        rmm::mr::device_memory_resource* mr,
        cudaStream_t stream)
    {
        uint32_t num_poly=fpos.size();
        uint32_t num_ring=rpos.size();
        uint32_t num_vertex=x.size();

        //std::cout<<"bounding_box_processor: num_poly="<<num_poly<<",num_ring="<<num_ring<<",num_vertex="<<num_vertex<<std::endl;

        auto exec_policy = rmm::exec_policy(stream);

        const uint32_t *d_ply_fpos=fpos.data<uint32_t>();
        const uint32_t *d_ply_rpos=rpos.data<uint32_t>();
        const T *d_ply_x=x.data<T>();
        const T *d_ply_y=y.data<T>();

        //compute bbox 

        rmm::device_buffer *db_first_ring_pos=new rmm::device_buffer(num_poly* sizeof(uint32_t),stream,mr);
        CUDF_EXPECTS(db_first_ring_pos!=nullptr, "error allocating memory for first ring positions"); 
        uint32_t *d_first_ring_pos=static_cast<uint32_t *>(db_first_ring_pos->data());

        rmm::device_buffer *db_temp_ring_pos=new rmm::device_buffer(num_poly* sizeof(uint32_t),stream,mr);
        CUDF_EXPECTS(db_temp_ring_pos!=nullptr, "error allocating temporal memory for first ring position array"); 
        uint32_t *d_temp_ring_pos=static_cast<uint32_t *>(db_temp_ring_pos->data());

        rmm::device_buffer *db_vertex_pid=new rmm::device_buffer(num_vertex* sizeof(uint32_t),stream,mr);
        CUDF_EXPECTS(db_vertex_pid!=nullptr, "error allocating temporal memory for vertex id array");
        uint32_t *d_vertex_pid=static_cast<uint32_t *>(db_vertex_pid->data());

        HANDLE_CUDA_ERROR( cudaMemset(d_first_ring_pos,0,num_poly*sizeof(uint32_t)) );
        HANDLE_CUDA_ERROR( cudaMemset(d_vertex_pid,0,num_vertex*sizeof(uint32_t)) );

if(0)
{
        printf("ring pos prefix sum\n"); 
        thrust::device_ptr<const uint32_t> d_fpos_ptr=thrust::device_pointer_cast(d_ply_fpos);
        thrust::copy(d_fpos_ptr,d_fpos_ptr+num_poly,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

        printf("vertex pos prefix sum\n"); 
        thrust::device_ptr<const uint32_t> d_rpos_ptr=thrust::device_pointer_cast(d_ply_rpos);
        thrust::copy(d_rpos_ptr,d_rpos_ptr+num_ring,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

        thrust::transform(exec_policy->on(stream),d_ply_fpos,d_ply_fpos+num_poly,d_temp_ring_pos,thrust::placeholders::_1-1);
        thrust::gather(exec_policy->on(stream),d_temp_ring_pos,d_temp_ring_pos+num_poly,d_ply_rpos,d_first_ring_pos);

        delete db_temp_ring_pos; db_temp_ring_pos=nullptr;

if(0)
{
        printf("prefix-sum numbers of points recorded at the last rings for all polygons\n");
        thrust::device_ptr<uint32_t> d_num_points_ptr=thrust::device_pointer_cast(d_first_ring_pos);
        thrust::copy(d_num_points_ptr,d_num_points_ptr+num_poly,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

        thrust::adjacent_difference(exec_policy->on(stream), d_first_ring_pos,d_first_ring_pos+num_poly,d_first_ring_pos);

if(0)
{
        printf("numbers of vertices for all rings\n"); 
        thrust::device_ptr<uint32_t> d_num_rings_ptr=thrust::device_pointer_cast(d_first_ring_pos);
        thrust::copy(d_num_rings_ptr,d_num_rings_ptr+num_poly,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}    
        thrust::exclusive_scan(exec_policy->on(stream),d_first_ring_pos,d_first_ring_pos+num_poly,d_first_ring_pos);
        thrust::scatter(exec_policy->on(stream),thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0)+num_poly,d_first_ring_pos,d_vertex_pid);
        thrust::inclusive_scan(exec_policy->on(stream),d_vertex_pid,d_vertex_pid+num_vertex,d_vertex_pid,thrust::maximum<int>());
if(0)
{
        printf("d_vertex_pid\n");
        thrust::device_ptr<uint32_t> d_vertex_pid_ptr=thrust::device_pointer_cast(d_vertex_pid);
        thrust::copy(d_vertex_pid_ptr,d_vertex_pid_ptr+num_vertex,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}


        rmm::device_buffer *db_bbox=new rmm::device_buffer(num_poly* sizeof(SBBox<T>),stream,mr);
        CUDF_EXPECTS(db_bbox!=nullptr, "error allocating memory for bboxes"); 
        SBBox<T> *d_p_bbox=static_cast<SBBox<T> *>(db_bbox->data());
        
        auto d_vertex_iter=thrust::make_zip_iterator(thrust::make_tuple(d_ply_x,d_ply_y));

        //reuse d_first_ring_pos to store sequential polygon index
        uint32_t num_bbox=thrust::reduce_by_key(exec_policy->on(stream),d_vertex_pid,d_vertex_pid+num_vertex,
            thrust::make_transform_iterator(d_vertex_iter,bbox_transformation<T>()),
            d_first_ring_pos,d_p_bbox,thrust::equal_to<uint32_t>(),bbox_reduction<T>()).first-d_first_ring_pos;
        std::cout<<"num_poly="<<num_poly<<",num_bbox="<<num_bbox<<std::endl;

        CUDF_EXPECTS(num_poly==num_bbox,"#of bbox after reduction should be the same as # of polys");

        delete db_first_ring_pos; db_first_ring_pos=nullptr;
        delete db_vertex_pid; db_vertex_pid=nullptr;

        std::unique_ptr<cudf::column> x1_col = cudf::make_numeric_column(
        cudf::data_type{cudf::experimental::type_to_id<T>()}, num_poly,cudf::mask_state::UNALLOCATED, stream, mr);
        T *x1=cudf::mutable_column_device_view::create(x1_col->mutable_view(), stream)->data<T>();
        assert(x1!=nullptr);

        std::unique_ptr<cudf::column> y1_col = cudf::make_numeric_column(
        cudf::data_type{cudf::experimental::type_to_id<T>()}, num_poly,cudf::mask_state::UNALLOCATED, stream, mr);
        T *y1=cudf::mutable_column_device_view::create(y1_col->mutable_view(), stream)->data<T>();
        assert(y1!=nullptr);

        std::unique_ptr<cudf::column> x2_col = cudf::make_numeric_column(
        cudf::data_type{cudf::experimental::type_to_id<T>()}, num_poly,cudf::mask_state::UNALLOCATED, stream, mr);
        T *x2=cudf::mutable_column_device_view::create(x2_col->mutable_view(), stream)->data<T>();
        assert(x2!=nullptr);

        std::unique_ptr<cudf::column> y2_col = cudf::make_numeric_column(
        cudf::data_type{cudf::experimental::type_to_id<T>()}, num_poly,cudf::mask_state::UNALLOCATED, stream, mr);
        T *y2=cudf::mutable_column_device_view::create(y2_col->mutable_view(), stream)->data<T>();
        assert(y2!=nullptr);

        auto out_bbox_iter=thrust::make_zip_iterator(thrust::make_tuple(x1,y1,x2,y2));
        thrust::transform(exec_policy->on(stream),d_p_bbox,d_p_bbox+num_bbox,out_bbox_iter,bbox2tuple<T>());

        delete db_bbox; db_bbox=nullptr;

        std::vector<std::unique_ptr<cudf::column>> bbox_cols;
        bbox_cols.push_back(std::move(x1_col));
        bbox_cols.push_back(std::move(y1_col));
        bbox_cols.push_back(std::move(x2_col));
        bbox_cols.push_back(std::move(y2_col));
        std::unique_ptr<cudf::experimental::table> destination_table = 
            std::make_unique<cudf::experimental::table>(std::move(bbox_cols));

        //std::cout<<"completing bounding_box_processor.................."<<std::endl;
        return destination_table;
}

    template<typename T, std::enable_if_t<!std::is_floating_point<T>::value >* = nullptr>
    std::unique_ptr<cudf::experimental::table> operator()(
        const cudf::column_view& fpos,const cudf::column_view& rpos,
        const cudf::column_view& x,const cudf::column_view& y,
        rmm::mr::device_memory_resource* mr,
        cudaStream_t stream)
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};

} //end anonymous namespace

namespace cuspatial {

std::unique_ptr<cudf::experimental::table> polygon_bbox(
    const cudf::column_view& fpos,const cudf::column_view& rpos,
    const cudf::column_view& x,const cudf::column_view& y)
{

    CUDF_EXPECTS(fpos.size()>0,"number of polygons must be greater than 0");
    CUDF_EXPECTS(rpos.size()>=fpos.size(),"number of rings must be no less than number of polygons");
    CUDF_EXPECTS(x.size()==y.size(),"numbers of vertices must be the same for both x and y columns");
    CUDF_EXPECTS(x.size()>=4*rpos.size(),"all rings must have at least 4 vertices");

    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    return cudf::experimental::type_dispatcher(x.type(),bounding_box_processor{},fpos,rpos,x,y,mr,stream);
}

}// namespace cuspatial
