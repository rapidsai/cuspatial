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
#include <random>
#include <algorithm>
#include <functional>

#include <gtest/gtest.h>
#include <utilities/legacy/error_utils.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

#include <utility/helper_thrust.cuh>
#include <utility/quadtree_thrust.cuh>
#include <utility/bbox_thrust.cuh>

#include <cuspatial/quadtree.hpp>
#include <cuspatial/bounding_box.hpp>
#include <cuspatial/spatial_jion.hpp>

struct PIPRefineTestLarge : public GdfTest 
{    
    uint32_t num_pnt=0;
    double *d_pnt_x=NULL,*d_pnt_y=NULL;
    std::unique_ptr<cudf::column> pnt_x,pnt_y;
    
    uint32_t num_poly=0,num_ring=0,num_vertex=0;
    uint32_t *d_poly_id=NULL,*d_poly_fpos=NULL,*d_poly_rpos=NULL;
    double *d_poly_x=NULL,*d_poly_y=NULL;
    
    std::unique_ptr<cudf::column> poly_fpos,poly_rpos,poly_x ,poly_y;
    
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    void setup_polygons()
    {
        uint32_t h_ply_fpos[]={1,2,3,4};
        uint32_t h_ply_rpos[]={4,10,14,19};
        double h_ply_x[] = {2.488450,1.333584,3.460720,2.488450,5.039823,5.561707,7.103516,7.190674,5.998939,5.039823,5.998939,5.573720,6.703534,5.998939,2.088115,1.034892,2.415080,3.208660,2.088115};
        double h_ply_y[] = {5.856625,5.008840,4.586599,5.856625,4.229242,1.825073,1.503906,4.025879,5.653384,4.229242,1.235638,0.197808,0.086693,1.235638,4.541529,3.530299,2.896937,3.745936,4.541529};
        num_poly=sizeof(h_ply_fpos)/sizeof(uint32_t);
        num_ring=sizeof(h_ply_rpos)/sizeof(uint32_t);
        num_vertex=sizeof(h_ply_x)/sizeof(double);
        assert(num_vertex==sizeof(h_ply_y)/sizeof(double));
        assert(num_vertex=h_ply_rpos[num_ring-1]); 	
        std::cout<<"setup_polygons:num_poly="<<this->num_poly<<std::endl;
        std::cout<<"setup_polygons:num_ring="<<this->num_ring<<std::endl;
        std::cout<<"setup_polygons:num_vertex="<<this->num_vertex<<std::endl;
   
        poly_fpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    		num_poly, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_fpos=cudf::mutable_column_device_view::create(poly_fpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_fpos!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_fpos, h_ply_fpos, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

        poly_rpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    		num_ring, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_rpos=cudf::mutable_column_device_view::create(poly_rpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_rpos!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_rpos, h_ply_rpos, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

        poly_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    		num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_x=cudf::mutable_column_device_view::create(poly_x->mutable_view(), stream)->data<double>();
        assert(d_poly_x!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_x, h_ply_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 

        poly_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    		num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_y=cudf::mutable_column_device_view::create(poly_y->mutable_view(), stream)->data<double>();
        assert(d_poly_y!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_y, h_ply_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );        
 }
  
 void setup_points(double x1,double y1,double x2,double y2,double scale,uint32_t num_levels,uint32_t min_size)
 {
    //9 leaf quadrants, the same as the small point dataset
    double quads[9][5]={{0,2,0,2},{3,4,0,1},{2,3,1,2},{4,6,0,2},{3,4,2,3},{2,3,3,4},{6,7,2,3},{7,8,3,4},{0,4,4,8}};
    std::vector<uint32_t> quad_pnt_nums(9);
    std::generate(quad_pnt_nums.begin(), quad_pnt_nums.end(), [&] () mutable { return min_size; });
    std::cout<<"quad_pnt_nums:"<<std::endl;
    std::copy(quad_pnt_nums.begin(),quad_pnt_nums.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
    num_pnt=std::accumulate(quad_pnt_nums.begin(), quad_pnt_nums.end(),0);
    std::cout<<"setup_points:num_pnt="<<this->num_pnt<<std::endl;    
   
    double h_pnt_x[num_pnt],h_pnt_y[num_pnt];
    std::seed_seq seed{0};
    std::mt19937 g(seed);
    int pos=0;
    
    for(uint32_t i=0;i<9;i++)
    {
 	 std::uniform_real_distribution<double> dist_x {quads[i][0], quads[i][1]};
 	 std::uniform_real_distribution<double> dist_y {quads[i][2], quads[i][3]};
         std::generate(h_pnt_x+pos, h_pnt_x+pos+quad_pnt_nums[i], [&] () mutable { return dist_x(g); });
         std::generate(h_pnt_y+pos, h_pnt_y+pos+quad_pnt_nums[i], [&] () mutable { return dist_y(g); });
         pos+=quad_pnt_nums[i];
    }
    assert(pos==num_pnt);
       
    pnt_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
    d_pnt_x=cudf::mutable_column_device_view::create(pnt_x->mutable_view(), stream)->data<double>();
    assert(d_pnt_x!=NULL);
    HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_x, h_pnt_x, num_pnt * sizeof(double), cudaMemcpyHostToDevice ) );    

    pnt_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
    d_pnt_y=cudf::mutable_column_device_view::create(pnt_y->mutable_view(), stream)->data<double>();
    assert(d_pnt_y!=NULL);    
    HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_y, h_pnt_y, num_pnt * sizeof(double), cudaMemcpyHostToDevice ) );     
 } 
 
void run_test(double x1,double y1,double x2,double y2,double scale,uint32_t num_levels,uint32_t min_size)
{       
     cudf::mutable_column_view pnt_x_view=pnt_x->mutable_view();
     cudf::mutable_column_view pnt_y_view=pnt_y->mutable_view();
     std::cout<<"run_test::num_pnt="<<pnt_x->size()<<std::endl;

     std::unique_ptr<cudf::experimental::table> quadtree= 
     	cuspatial::quadtree_on_points(pnt_x_view,pnt_y_view,x1,y1,x2,y2, scale,num_levels, min_size);
     std::cout<<"run_test: quadtree num cols="<<quadtree->view().num_columns()<<std::endl;
     
     std::unique_ptr<cudf::experimental::table> bbox_tbl=
     	cuspatial::polygon_bbox(poly_fpos->view(),poly_rpos->view(),poly_x->view(),poly_y->view()); 
     std::cout<<"polygon bbox="<<bbox_tbl->view().num_rows()<<std::endl;
     
     const cudf::table_view quad_view=quadtree->view();
     const cudf::table_view bbox_view=bbox_tbl->view();
     
     std::unique_ptr<cudf::experimental::table> pq_pair_tbl=cuspatial::quad_bbox_join(
         quad_view,bbox_view,x1,y1,x2,y2, scale,num_levels, min_size);   
     std::cout<<"polygon/quad num pair="<<pq_pair_tbl->view().num_columns()<<std::endl;
 
     const cudf::table_view pq_pair_view=pq_pair_tbl->view();
     const cudf::table_view pnt_view({pnt_x_view,pnt_y_view});
 
     std::unique_ptr<cudf::experimental::table> pip_pair_tbl=cuspatial::pip_refine(
         pq_pair_view,quad_view,pnt_view,
         poly_fpos->view(),poly_rpos->view(),poly_x->view(),poly_y->view());   
     std::cout<<"polygon/point num pair="<<pip_pair_tbl->view().num_columns()<<std::endl;
}
 
};

TEST_F(PIPRefineTestLarge, test)
{
    const uint32_t num_levels=3;
    const uint32_t min_size=200;
    double scale=1.0;
    double x1=0,x2=8,y1=0,y2=8;
 
    this->setup_polygons();
    this->setup_points(x1,y1,x2,y2,scale,num_levels,min_size);
    std::cout<<"running test_point_large..........."<<std::endl;
    this->run_test(x1,y1,x2,y2,scale,num_levels,min_size);
}

