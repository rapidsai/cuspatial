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

struct PIPRefineTestSmall : public GdfTest 
{    
    uint32_t num_pnt=0;
    uint32_t * d_pnt_id=NULL;
    double *d_pnt_x=NULL,*d_pnt_y=NULL;
    std::unique_ptr<cudf::column> pnt_id,pnt_x,pnt_y;

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

    double h_pnt_x[]={1.9804558865545805, 0.1895259128530169, 1.2591725716781235, 0.8178039499335275, 0.48171647380517046, 1.3890664414691907, 0.2536015260915061, 3.1907684812039956, 3.028362149164369, 3.918090468102582, 3.710910700915217, 3.0706987088385853, 3.572744183805594, 3.7080407833612004, 3.70669993057843, 3.3588457228653024, 2.0697434332621234, 2.5322042870739683, 2.175448214220591, 2.113652420701984, 2.520755151373394, 2.9909779614491687, 2.4613232527836137, 4.975578758530645, 4.07037627210835, 4.300706849071861, 4.5584381091040616, 4.822583857757069, 4.849847745942472, 4.75489831780737, 4.529792124514895, 4.732546857961497, 3.7622247877537456, 3.2648444465931474, 3.01954722322135, 3.7164018490892348, 3.7002781846945347, 2.493975723955388, 2.1807636574967466, 2.566986568683904, 2.2006520196663066, 2.5104987015171574, 2.8222482218882474, 2.241538022180476, 2.3007438625108882, 6.0821276168848994, 6.291790729917634, 6.109985464455084, 6.101327777646798, 6.325158445513714, 6.6793884701899, 6.4274219368674315, 6.444584786789386, 7.897735998643542, 7.079453687660189, 7.430677191305505, 7.5085184104988, 7.886010001346151, 7.250745898479374, 7.769497359206111, 1.8703303641352362, 1.7015273093278767, 2.7456295127617385, 2.2065031771469, 3.86008672302403, 1.9143371250907073, 3.7176098065039747, 0.059011873032214, 3.1162712022943757, 2.4264509160270813, 3.154282922203257};
    num_pnt=sizeof(h_pnt_x)/sizeof(double);
    double h_pnt_y[]={1.3472225743317712, 0.5431061133894604, 0.1448705855995005, 0.8138440641113271, 1.9022922214961997, 1.5177694304735412, 1.8762161698642947, 0.2621847215928189, 0.027638405909631958, 0.3338651960183463, 0.9937713340192049, 0.9376313558467103, 0.33184908855075124, 0.09804238103130436, 0.7485845679979923, 0.2346381514128677, 1.1809465376402173, 1.419555755682142, 1.2372448404986038, 1.2774712415624014, 1.902015274420646, 1.2420487904041893, 1.0484414482621331, 0.9606291981013242, 1.9486902798139454, 0.021365525588281198, 1.8996548860019926, 0.3234041700489503, 1.9531893897409585, 0.7800065259479418, 1.942673409259531, 0.5659923375279095, 2.8709552313924487, 2.693039435509084, 2.57810040095543, 2.4612194182614333, 2.3345952955903906, 3.3999020934055837, 3.2296461832828114, 3.6607732238530897, 3.7672478678985257, 3.0668114607133137, 3.8159308233351266, 3.8812819070357545, 3.6045900851589048, 2.5470532680258002, 2.983311357415729, 2.2235950639628523, 2.5239201807166616, 2.8765450351723674, 2.5605928243991434, 2.9754616970668213, 2.174562817047202, 3.380784914178574, 3.063690547962938, 3.380489849365283, 3.623862886287816, 3.538128217886674, 3.4154469467473447, 3.253257011908445, 4.209727933188015, 7.478882372510933, 7.474216636277054, 6.896038613284851, 7.513564222799629, 6.885401350515916, 6.194330707468438, 5.823535317960799, 6.789029097334483, 5.188939408363776, 5.788316610960881};
    assert(sizeof(h_pnt_y)/sizeof(double)==num_pnt);
    std::cout<<"setup_points:num_pnt="<<this->num_pnt<<std::endl;
    
    pnt_id = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
     	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
    d_pnt_id=cudf::mutable_column_device_view::create(pnt_id->mutable_view(), stream)->data<uint32_t>();
    assert(d_pnt_id!=NULL);
    thrust::sequence(thrust::device,d_pnt_id,d_pnt_id+num_pnt);
  
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
     uint32_t local_num_pnt=pnt_x_view.size();
     std::cout<<"run_test::local_num_pnt="<<num_pnt<<std::endl;

     std::unique_ptr<cudf::experimental::table> quadtree= 
     	cuspatial::quadtree_on_points(pnt_x_view,pnt_y_view,x1,y1,x2,y2, scale,num_levels, min_size);
     uint32_t num_quad_nodes=quadtree->view().num_rows();
     std::cout<<"run_test: # quadtree nodes="<<num_quad_nodes<<std::endl;
 
     std::unique_ptr<cudf::experimental::table> bbox_tbl=
     	cuspatial::polygon_bbox(poly_fpos->view(),poly_rpos->view(),poly_x->view(),poly_y->view()); 
     uint32_t local_num_poly=bbox_tbl->view().num_rows();
     std::cout<<"run_test: # polygon bbox="<<local_num_poly<<std::endl;
     
     const cudf::table_view quad_view=quadtree->view();
     const cudf::table_view bbox_view=bbox_tbl->view();
     
     std::unique_ptr<cudf::experimental::table> pq_pair_tbl=cuspatial::quad_bbox_join(
         quad_view,bbox_view,x1,y1,x2,y2, scale,num_levels, min_size);   
     uint32_t  num_pq_pairs=pq_pair_tbl->view().num_rows();
     std::cout<<"run_test: # polygon/quad pairs="<<num_pq_pairs<<std::endl;
 
     const cudf::table_view pq_pair_view=pq_pair_tbl->view();
     const cudf::table_view pnt_view({pnt_x_view,pnt_y_view});
 
     std::unique_ptr<cudf::experimental::table> pip_pair_tbl=cuspatial::pip_refine(
         pq_pair_view,quad_view,pnt_view,
         poly_fpos->view(),poly_rpos->view(),poly_x->view(),poly_y->view());   
     uint32_t num_pp_pairs=pip_pair_tbl->view().num_rows();
     std::cout<<"run_test: # polygon/point pair="<<num_pp_pairs<<std::endl;

if(0)
{        
        //note that pnt x/y have been sorted 
        double *h_pnt_x=new double[num_pnt];
        double *h_pnt_y=new double[num_pnt];
        HANDLE_CUDA_ERROR( cudaMemcpy(h_pnt_x, d_pnt_x,num_pnt * sizeof(double), cudaMemcpyDeviceToHost ) );    
        HANDLE_CUDA_ERROR( cudaMemcpy(h_pnt_y, d_pnt_y,num_pnt * sizeof(double), cudaMemcpyDeviceToHost ) );    
        
        for(uint32_t i=0;i<num_pnt;i++)
             printf("%10d, %15.5f, %15.5f\n",i,h_pnt_x[i],h_pnt_y[i]);
        printf("\n\n");
       
        const uint32_t * d_pp_poly_idx=pip_pair_tbl->view().column(0).data<uint32_t>();
        const uint32_t * d_pp_pnt_idx=pip_pair_tbl->view().column(1).data<uint32_t>();      
        uint32_t *h_pp_poly_idx=new uint32_t[num_pp_pairs];
        uint32_t *h_pp_pnt_idx=new uint32_t[num_pp_pairs];
        assert(h_pp_poly_idx!=NULL && h_pp_pnt_idx!=NULL);
 	HANDLE_CUDA_ERROR( cudaMemcpy( h_pp_poly_idx, d_pp_poly_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
 	HANDLE_CUDA_ERROR( cudaMemcpy( h_pp_pnt_idx, d_pp_pnt_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) );   
         
        for(uint32_t i=0;i<num_pp_pairs;i++)
             printf("%10d, %15.5f, %15.5f, %10d\n",i,h_pnt_x[h_pp_pnt_idx[i]],h_pnt_y[h_pp_pnt_idx[i]],h_pp_poly_idx[i]);
}
    } 
};

TEST_F(PIPRefineTestSmall, test_point_small)
{
     const uint32_t num_levels=3;
     const uint32_t min_size=12;
     double scale=1.0;
     double x1=0,x2=8,y1=0,y2=8;
     
     this->setup_polygons();
     this->setup_points(x1,y1,x2,y2,scale,num_levels,min_size);
     std::cout<<"running test_point_small..........."<<std::endl;
     this->run_test(x1,y1,x2,y2,scale,num_levels,min_size);
}

