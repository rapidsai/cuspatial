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

TEST_F(BoundingBoxTest, test_empty)
{
    cudf::column fpos_col,rpos_col;
    cudf::column x_col,y_col;

    EXPECT_THROW (cuspatial::polygon_bbox(fpos_col,rpos_col,x_col,y_col),cudf::logic_error);    
}

TEST_F(BoundingBoxTest, test_one)
{
    uint32_t poly_fpos[]={1};
    uint32_t poly_rpos[]={4};
    double poly_x[] = {2.488450,1.333584,3.460720,2.488450};
    double poly_y[] = {5.856625,5.008840,4.586599,5.856625};

    uint32_t num_poly=sizeof(poly_fpos)/sizeof(uint32_t);
    uint32_t num_ring=sizeof(poly_rpos)/sizeof(uint32_t);
    uint32_t num_vertex=sizeof(poly_x)/sizeof(double);   
    std::cout<<"num_poly="<<num_poly<<",num_ring="<<num_ring<<",num_vertex="<<num_vertex<<std::endl;

    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    
    std::unique_ptr<cudf::column> fpos_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    	num_poly, cudf::mask_state::UNALLOCATED, stream, mr );      
    uint32_t *d_p_fpos=cudf::mutable_column_device_view::create(fpos_col->mutable_view(), stream)->data<uint32_t>();
    assert(d_p_fpos!=nullptr);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_fpos, poly_fpos, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> rpos_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    	num_ring, cudf::mask_state::UNALLOCATED, stream, mr );      
    uint32_t *d_p_rpos=cudf::mutable_column_device_view::create(rpos_col->mutable_view(), stream)->data<uint32_t>();
    assert(d_p_rpos!=nullptr);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_rpos, poly_rpos, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> x_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    	num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
    double *d_p_x=cudf::mutable_column_device_view::create(x_col->mutable_view(), stream)->data<double>();
    assert(d_p_x!=nullptr);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_x, poly_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> y_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
        num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
    double *d_p_y=cudf::mutable_column_device_view::create(y_col->mutable_view(), stream)->data<double>();
    assert(d_p_y!=nullptr);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_y, poly_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 

    //GPU computation
    std::unique_ptr<cudf::experimental::table> bbox_tbl=cuspatial::polygon_bbox(
        *fpos_col,*rpos_col,*x_col,*y_col);

    CUDF_EXPECTS(bbox_tbl->view().num_columns()==4, "bbox table must have 4 columns");
    CUDF_EXPECTS((uint32_t)(bbox_tbl->num_rows())==num_poly,"resutling #of bounding boxes must be the same as # of polygons");    
  
    const double *d_rx1=bbox_tbl->get_column(0).view().data<double>();
    const double *d_ry1=bbox_tbl->get_column(1).view().data<double>();
    const double *d_rx2=bbox_tbl->get_column(2).view().data<double>();
    const double *d_ry2=bbox_tbl->get_column(3).view().data<double>();
  
    double *h_rx1=new double[num_poly];
    double *h_ry1=new double[num_poly];
    double *h_rx2=new double[num_poly];
    double *h_ry2=new double[num_poly];
    assert(h_rx1!=nullptr && h_ry1!=nullptr && h_rx2!=nullptr && h_ry2!=nullptr);

    EXPECT_EQ(cudaMemcpy(h_rx1,d_rx1,num_poly*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_ry1,d_ry1,num_poly*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_rx2,d_rx2,num_poly*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_ry2,d_ry2,num_poly*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);

    EXPECT_NEAR(h_rx1[0],1.333584, 1e-9);
    EXPECT_NEAR(h_ry1[0],4.586599, 1e-9);
    EXPECT_NEAR(h_rx2[0],3.460720, 1e-9);
    EXPECT_NEAR(h_ry2[0],5.856625, 1e-9);
}

TEST_F(BoundingBoxTest, test_small)
{
    uint32_t poly_fpos[]={1,2,3,4};
    uint32_t poly_rpos[]={4,10,14,19};
    double poly_x[] = {2.488450,1.333584,3.460720,2.488450,5.039823,5.561707,7.103516,7.190674,5.998939,5.039823,5.998939,5.573720,6.703534,5.998939,2.088115,1.034892,2.415080,3.208660,2.088115};
    double poly_y[] = {5.856625,5.008840,4.586599,5.856625,4.229242,1.825073,1.503906,4.025879,5.653384,4.229242,1.235638,0.197808,0.086693,1.235638,4.541529,3.530299,2.896937,3.745936,4.541529};
    
    uint32_t num_poly=sizeof(poly_fpos)/sizeof(uint32_t);
    uint32_t num_ring=sizeof(poly_rpos)/sizeof(uint32_t);
    uint32_t num_vertex=sizeof(poly_x)/sizeof(double);   
    std::cout<<"num_poly="<<num_poly<<",num_ring="<<num_ring<<",num_vertex="<<num_vertex<<std::endl;

    CUDF_EXPECTS(num_vertex==sizeof(poly_y)/sizeof(double),"x/y should have same length");
    CUDF_EXPECTS(num_vertex=poly_rpos[num_ring-1],"# of vertex should be the same as the last element of the ring array");
    
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    
    std::unique_ptr<cudf::column> fpos_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    	num_poly, cudf::mask_state::UNALLOCATED, stream, mr );      
    uint32_t *d_p_fpos=cudf::mutable_column_device_view::create(fpos_col->mutable_view(), stream)->data<uint32_t>();
    assert(d_p_fpos!=nullptr);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_fpos, poly_fpos, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> rpos_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    	num_ring, cudf::mask_state::UNALLOCATED, stream, mr );      
    uint32_t *d_p_rpos=cudf::mutable_column_device_view::create(rpos_col->mutable_view(), stream)->data<uint32_t>();
    assert(d_p_rpos!=nullptr);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_rpos, poly_rpos, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> x_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    	num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
    double *d_p_x=cudf::mutable_column_device_view::create(x_col->mutable_view(), stream)->data<double>();
    assert(d_p_x!=nullptr);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_x, poly_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 

    std::unique_ptr<cudf::column> y_col = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
        num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
    double *d_p_y=cudf::mutable_column_device_view::create(y_col->mutable_view(), stream)->data<double>();
    assert(d_p_y!=nullptr);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_y, poly_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 

    //GPU computation
    std::unique_ptr<cudf::experimental::table> bbox_tbl=cuspatial::polygon_bbox(
        *fpos_col,*rpos_col,*x_col,*y_col);

    CUDF_EXPECTS(bbox_tbl->view().num_columns()==4, "bbox table must have 4 columns");
    CUDF_EXPECTS((uint32_t)(bbox_tbl->num_rows())==num_poly,"resutling #of bounding boxes must be the same as # of polygons");

    //CPU computation
    double *c_rx1=new double[num_poly];
    double *c_ry1=new double[num_poly];
    double *c_rx2=new double[num_poly];
    double *c_ry2=new double[num_poly];
    assert(c_rx1!=nullptr && c_ry1!=nullptr && c_rx2!=nullptr && c_ry2!=nullptr);

    for(uint32_t fid=0;fid<num_poly;fid++)
    {
        uint32_t r_f = (0 == fid) ? 0 : poly_fpos[fid-1];
        uint32_t r_t=poly_fpos[fid];

        uint32_t n = (r_f==0)?0:poly_rpos[r_f-1];
        c_rx1[fid]=c_rx2[fid]=poly_x[n];
        c_ry1[fid]=c_ry2[fid]=poly_y[n];
        
        for (uint32_t r = r_f; r < r_t; r++) //for each ring
        {
            uint32_t m = (r==0)?0:poly_rpos[r-1];
            for (;m < poly_rpos[r]-1; m++) //for each line segment
            {
                if( c_rx1[fid]>poly_x[m]) c_rx1[fid]=poly_x[m];
                if( c_rx2[fid]<poly_x[m]) c_rx2[fid]=poly_x[m];
                if( c_ry1[fid]>poly_y[m]) c_ry1[fid]=poly_y[m];
                if( c_ry2[fid]<poly_y[m]) c_ry2[fid]=poly_y[m];
            }
        }
    }

    const double *d_rx1=bbox_tbl->get_column(0).view().data<double>();
    const double *d_ry1=bbox_tbl->get_column(1).view().data<double>();
    const double *d_rx2=bbox_tbl->get_column(2).view().data<double>();
    const double *d_ry2=bbox_tbl->get_column(3).view().data<double>();

if(0)
{

    std::cout<<"x1:"<<std::endl;
    thrust::device_ptr<const double> x1_ptr=thrust::device_pointer_cast(d_rx1);
    thrust::copy(x1_ptr,x1_ptr+num_poly,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"y1:"<<std::endl;
    thrust::device_ptr<const double> y1_ptr=thrust::device_pointer_cast(d_ry1);
    thrust::copy(y1_ptr,y1_ptr+num_poly,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"x2:"<<std::endl;
    thrust::device_ptr<const double> x2_ptr=thrust::device_pointer_cast(d_rx2);
    thrust::copy(x2_ptr,x2_ptr+num_poly,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;

    std::cout<<"y2:"<<std::endl;
    thrust::device_ptr<const double> y2_ptr=thrust::device_pointer_cast(d_ry2);
    thrust::copy(y2_ptr,y2_ptr+num_poly,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;
}

    double *h_rx1=new double[num_poly];
    double *h_ry1=new double[num_poly];
    double *h_rx2=new double[num_poly];
    double *h_ry2=new double[num_poly];
    assert(h_rx1!=nullptr && h_ry1!=nullptr && h_rx2!=nullptr && h_ry2!=nullptr);

    EXPECT_EQ(cudaMemcpy(h_rx1,d_rx1,num_poly*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_ry1,d_ry1,num_poly*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_rx2,d_rx2,num_poly*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_ry2,d_ry2,num_poly*sizeof(double),cudaMemcpyDeviceToHost),cudaSuccess);

    for(uint32_t i=0;i<num_poly;i++)
    {
        EXPECT_NEAR(c_rx1[i], h_rx1[i], 1e-9);
        EXPECT_NEAR(c_ry1[i], h_ry1[i], 1e-9);
        EXPECT_NEAR(c_rx2[i], h_rx2[i], 1e-9);
        EXPECT_NEAR(c_ry2[i], h_ry2[i], 1e-9);
    }
    std::cout<<"bounding_box_test: verified"<<std::endl;

    delete[] c_rx1; 
    delete[] c_ry1;
    delete[] c_rx2;
    delete[] c_ry2;

    delete[] h_rx1; 
    delete[] h_ry1;
    delete[] h_rx2;
    delete[] h_ry2;    

}


