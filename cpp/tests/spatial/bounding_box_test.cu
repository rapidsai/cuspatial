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
#include <string>

#include <gtest/gtest.h>
#include <utilities/legacy/error_utils.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

#include <utility/helper_thrust.cuh>
#include <cuspatial/bounding_box.hpp>

struct BoundingBoxTest : public GdfTest 
{
 
};

TEST_F(BoundingBoxTest, test1)
{
    uint32_t ply_fpos[]={1,2,3,4};
    uint32_t ply_rpos[]={4,10,14,19};
    double ply_x[] = {2.488450,1.333584,3.460720,2.488450,5.039823,5.561707,7.103516,7.190674,5.998939,5.039823,5.998939,5.573720,6.703534,5.998939,2.088115,1.034892,2.415080,3.208660,2.088115};
    double ply_y[] = {5.856625,5.008840,4.586599,5.856625,4.229242,1.825073,1.503906,4.025879,5.653384,4.229242,1.235638,0.197808,0.086693,1.235638,4.541529,3.530299,2.896937,3.745936,4.541529};
    
    uint32_t num_poly=sizeof(ply_fpos)/sizeof(uint32_t);
    uint32_t num_ring=sizeof(ply_rpos)/sizeof(uint32_t);
    uint32_t num_vertex=sizeof(ply_x)/sizeof(double);   
    std::cout<<"num_poly="<<num_poly<<",num_ring="<<num_ring<<",num_vertex="<<num_vertex<<std::endl;

    CUDF_EXPECTS(num_vertex==sizeof(ply_y)/sizeof(double),"x/y should have same length");
    CUDF_EXPECTS(num_vertex=ply_rpos[num_ring-1],"# of vertex should be the same as the last postion");
    
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    
    std::unique_ptr<cudf::column> fpos_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    	num_poly, cudf::mask_state::UNALLOCATED, stream, mr );      
    uint32_t *d_p_fpos=cudf::mutable_column_device_view::create(fpos_col->mutable_view(), stream)->data<uint32_t>();
    assert(d_p_fpos!=NULL);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_fpos, ply_fpos, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> rpos_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    	num_ring, cudf::mask_state::UNALLOCATED, stream, mr );      
    uint32_t *d_p_rpos=cudf::mutable_column_device_view::create(rpos_col->mutable_view(), stream)->data<uint32_t>();
    assert(d_p_rpos!=NULL);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_rpos, ply_rpos, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> x_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    	num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
    double *d_p_x=cudf::mutable_column_device_view::create(x_col->mutable_view(), stream)->data<double>();
    assert(d_p_x!=NULL);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_x, ply_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> y_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    	num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
    double *d_p_y=cudf::mutable_column_device_view::create(y_col->mutable_view(), stream)->data<double>();
    assert(d_p_y!=NULL);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_y, ply_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 
   
   std::unique_ptr<cudf::experimental::table> bbox_tbl=cuspatial::polygon_bbox(
	*fpos_col,*rpos_col,*x_col,*y_col);
   
   std::cout<<"num cols="<<bbox_tbl->view().num_columns()<<std::endl; 
   const double *rx1=bbox_tbl->get_column(0).view().data<double>();
   const double *ry1=bbox_tbl->get_column(1).view().data<double>();
   const double *rx2=bbox_tbl->get_column(2).view().data<double>();
   const double *ry2=bbox_tbl->get_column(3).view().data<double>();
   CUDF_EXPECTS((uint32_t)(bbox_tbl->num_rows())==num_poly,"resutling #of bounding boxes must be the same as # of polygons");
   
   std::cout<<"x1:"<<std::endl;
   thrust::device_ptr<const double> x1_ptr=thrust::device_pointer_cast(rx1);
   thrust::copy(x1_ptr,x1_ptr+num_poly,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;

   std::cout<<"y1:"<<std::endl;
   thrust::device_ptr<const double> y1_ptr=thrust::device_pointer_cast(ry1);
   thrust::copy(y1_ptr,y1_ptr+num_poly,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;

   std::cout<<"x2:"<<std::endl;
   thrust::device_ptr<const double> x2_ptr=thrust::device_pointer_cast(rx2);
   thrust::copy(x2_ptr,x2_ptr+num_poly,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;

   std::cout<<"y2:"<<std::endl;
   thrust::device_ptr<const double> y2_ptr=thrust::device_pointer_cast(ry2);
   thrust::copy(y2_ptr,y2_ptr+num_poly,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;
}


