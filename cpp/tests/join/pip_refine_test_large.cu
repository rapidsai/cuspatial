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

#include <ogrsf_frmts.h>

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
    uint32_t num_pnts=0;
    double *h_pnt_x=nullptr,*h_pnt_y=nullptr;    
    std::unique_ptr<cudf::column> pnt_x,pnt_y;

    uint32_t num_polys=0,num_rings=0,num_vertices=0;
    std::vector<OGRGeometry *> h_polygon_vec;
    std::unique_ptr<cudf::column> poly_fpos,poly_rpos,poly_x ,poly_y;
       
    uint32_t num_pp_pairs=0;
    uint32_t *h_pp_pnt_idx=nullptr,*h_pp_poly_idx=nullptr;
    
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    void setup_polygons()
    {
        uint32_t h_poly_fpos[]={1,2,3,4};
        uint32_t h_poly_rpos[]={4,10,14,19};
        double h_poly_x[] = {2.488450,1.333584,3.460720,2.488450,5.039823,5.561707,7.103516,7.190674,5.998939,5.039823,5.998939,5.573720,6.703534,5.998939,2.088115,1.034892,2.415080,3.208660,2.088115};
        double h_poly_y[] = {5.856625,5.008840,4.586599,5.856625,4.229242,1.825073,1.503906,4.025879,5.653384,4.229242,1.235638,0.197808,0.086693,1.235638,4.541529,3.530299,2.896937,3.745936,4.541529};

        this->num_polys=sizeof(h_poly_fpos)/sizeof(uint32_t);
        this->num_rings=sizeof(h_poly_rpos)/sizeof(uint32_t);
        this->num_vertices=sizeof(h_poly_x)/sizeof(double);
        assert(this->num_vertices==sizeof(h_poly_y)/sizeof(double));
        assert(this->num_vertices=h_poly_rpos[num_rings-1]);

        std::cout<<"setup_polygons:num_polys="<<this->num_polys<<std::endl;
        std::cout<<"setup_polygons:num_rings="<<this->num_rings<<std::endl;
        std::cout<<"setup_polygons:num_vertices="<<this->num_vertices<<std::endl;

        this->poly_fpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
            this->num_polys, cudf::mask_state::UNALLOCATED, stream, mr );      
        uint32_t *d_poly_fpos=cudf::mutable_column_device_view::create(poly_fpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_fpos!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_fpos, h_poly_fpos, this->num_polys * sizeof(uint32_t), cudaMemcpyHostToDevice ) );

        this->poly_rpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32},
            num_rings, cudf::mask_state::UNALLOCATED, stream, mr );      
        uint32_t * d_poly_rpos=cudf::mutable_column_device_view::create(this->poly_rpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_rpos!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_rpos, h_poly_rpos, this->num_rings * sizeof(uint32_t), cudaMemcpyHostToDevice ) );

        this->poly_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64},
            num_vertices, cudf::mask_state::UNALLOCATED, stream, mr );
        double *d_poly_x=cudf::mutable_column_device_view::create(this->poly_x->mutable_view(), stream)->data<double>();
        assert(d_poly_x!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_x, h_poly_x, this->num_vertices * sizeof(double), cudaMemcpyHostToDevice ) );

        this->poly_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            num_vertices, cudf::mask_state::UNALLOCATED, stream, mr );      
        double *d_poly_y=cudf::mutable_column_device_view::create(this->poly_y->mutable_view(), stream)->data<double>();
        assert(d_poly_y!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_y, h_poly_y, this->num_vertices * sizeof(double), cudaMemcpyHostToDevice ) );

        //populte h_polygon_vec for verification later
        this->h_polygon_vec.clear();
        uint32_t rc=0,vc=0;
        for(uint32_t fid=0;fid<num_polys;fid++)
        {
            uint32_t r_f = (0 == fid) ? 0 : h_poly_fpos[fid-1];
            uint32_t r_t=h_poly_fpos[fid];
            OGRPolygon *polygon=(OGRPolygon*)OGRGeometryFactory::createGeometry(wkbPolygon);
            for (uint32_t r = r_f; r < r_t; r++) //for each ring
            {
                OGRLineString *ls=(OGRLinearRing*)OGRGeometryFactory::createGeometry(wkbLinearRing);
                uint32_t m = (r==0)?0:h_poly_rpos[r-1];
                for (;m < h_poly_rpos[r]; m++) //for each line segment
                {
                    ls->addPoint(h_poly_x[m],h_poly_y[m]);
                    vc++;
                }
                polygon->addRing(ls);
                rc++;
            }
            this->h_polygon_vec.push_back(polygon);
        }
        std::cout<<"rc="<<rc<<" vc="<<vc<<std::endl;
    }

    void setup_points(double x1,double y1,double x2,double y2,double scale,uint32_t num_levels,uint32_t min_size)
    {
        //9 leaf quadrants, the same as the small point dataset
        const uint32_t num_quads=9;
        
        double quads[num_quads][4]={{0,2,0,2},{3,4,0,1},{2,3,1,2},{4,6,0,2},{3,4,2,3},{2,3,3,4},{6,7,2,3},{7,8,3,4},{0,4,4,8}};

        //for each quadrant, generate min_size points in the quadrant 
        //min_size should be set to be large than 32 to test the correctness of the two CUDA kernels for spatial refinement
        std::vector<uint32_t> quad_pnt_nums(9);
        std::generate(quad_pnt_nums.begin(), quad_pnt_nums.end(), [&] () mutable { return min_size; });

        std::copy(quad_pnt_nums.begin(),quad_pnt_nums.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
        num_pnts=std::accumulate(quad_pnt_nums.begin(), quad_pnt_nums.end(),0);
        std::cout<<"setup_points:num_pnts="<<this->num_pnts<<std::endl;

        this->h_pnt_x=new double[num_pnts];
        this->h_pnt_y=new double[num_pnts];
        assert(this->h_pnt_x!=nullptr && this->h_pnt_y!=nullptr);

        std::seed_seq seed{time(0)};
        std::mt19937 g(seed);
        uint32_t pos=0;

        for(uint32_t i=0;i<num_quads;i++)
        {
            std::uniform_real_distribution<double> dist_x {quads[i][0], quads[i][1]};
            std::uniform_real_distribution<double> dist_y {quads[i][2], quads[i][3]};
            std::generate(h_pnt_x+pos, h_pnt_x+pos+quad_pnt_nums[i], [&] () mutable { return dist_x(g); });
            std::generate(h_pnt_y+pos, h_pnt_y+pos+quad_pnt_nums[i], [&] () mutable { return dist_y(g); });
            pos+=quad_pnt_nums[i];
        }
        assert(pos==num_pnts);

        pnt_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            num_pnts, cudf::mask_state::UNALLOCATED, stream, mr );      
        double *d_pnt_x=cudf::mutable_column_device_view::create(pnt_x->mutable_view(), stream)->data<double>();
        assert(d_pnt_x!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_x, this->h_pnt_x, num_pnts * sizeof(double), cudaMemcpyHostToDevice ) );

        pnt_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            num_pnts, cudf::mask_state::UNALLOCATED, stream, mr );      
        double *d_pnt_y=cudf::mutable_column_device_view::create(pnt_y->mutable_view(), stream)->data<double>();
        assert(d_pnt_y!=nullptr);    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_y, this->h_pnt_y, num_pnts * sizeof(double), cudaMemcpyHostToDevice ) );
    }

    void run_test(double x1,double y1,double x2,double y2,double scale,uint32_t num_levels,uint32_t min_size)
    {
        cudf::mutable_column_view pnt_x_view=pnt_x->mutable_view();
        cudf::mutable_column_view pnt_y_view=pnt_y->mutable_view();
        std::cout<<"run_test::num_pnts="<<pnt_x->size()<<std::endl;

        std::unique_ptr<cudf::experimental::table> quadtree= 
            cuspatial::quadtree_on_points(pnt_x_view,pnt_y_view,x1,y1,x2,y2, scale,num_levels, min_size);

        double * d_pnt_x=pnt_x_view.data<double>();
        double * d_pnt_y=pnt_y_view.data<double>();
        HANDLE_CUDA_ERROR( cudaMemcpy(h_pnt_x, d_pnt_x,num_pnts * sizeof(double), cudaMemcpyDeviceToHost ) );
        HANDLE_CUDA_ERROR( cudaMemcpy(h_pnt_y, d_pnt_y,num_pnts * sizeof(double), cudaMemcpyDeviceToHost ) );


        std::unique_ptr<cudf::experimental::table> bbox_tbl=
            cuspatial::polygon_bbox(poly_fpos->view(),poly_rpos->view(),poly_x->view(),poly_y->view());

        const cudf::table_view quad_view=quadtree->view();
        const cudf::table_view bbox_view=bbox_tbl->view();

        std::unique_ptr<cudf::experimental::table> pq_pair_tbl=cuspatial::quad_bbox_join(
            quad_view,bbox_view,x1,y1,x2,y2, scale,num_levels, min_size);

        const cudf::table_view pq_pair_view=pq_pair_tbl->view();
        const cudf::table_view pnt_view({pnt_x_view,pnt_y_view});

        std::unique_ptr<cudf::experimental::table> pip_pair_tbl=cuspatial::pip_refine(
            pq_pair_view,quad_view,pnt_view,
            poly_fpos->view(),poly_rpos->view(),poly_x->view(),poly_y->view());

        cudf::table_view  pip_pair_view= pip_pair_tbl->view();
        this->num_pp_pairs=pip_pair_view.num_rows();
        std::cout<<"run_test: # polygon/point pair="<<num_pp_pairs<<std::endl;
        CUDF_EXPECTS(pip_pair_view.num_columns()==2,"a polygon-quadrant pair table must have 2 columns");

        const uint32_t * d_pp_poly_idx=pip_pair_tbl->view().column(0).data<uint32_t>();
        const uint32_t * d_pp_pnt_idx=pip_pair_tbl->view().column(1).data<uint32_t>();

        this->h_pp_pnt_idx=new uint32_t[this->num_pp_pairs];
        this->h_pp_poly_idx=new uint32_t[this->num_pp_pairs];
        assert(this->h_pp_pnt_idx!=nullptr && this->h_pp_poly_idx!=nullptr);

        HANDLE_CUDA_ERROR( cudaMemcpy(h_pp_poly_idx, d_pp_poly_idx,num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost ) );
        HANDLE_CUDA_ERROR( cudaMemcpy(h_pp_pnt_idx, d_pp_pnt_idx,num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost ) );
    }

    void run_verify()
    {

if(0)
{   
        std::cout<<"polygon index"<<std::endl;
        thrust::copy(h_pp_poly_idx,h_pp_poly_idx+num_pp_pairs,std::ostream_iterator<const uint32_t>(std::cout, " "));std::cout<<std::endl;

        std::cout<<"point index"<<std::endl;
        thrust::copy(h_pp_pnt_idx,h_pp_pnt_idx+num_pp_pairs,std::ostream_iterator<const uint32_t>(std::cout, " "));std::cout<<std::endl;
}

        thrust::sort_by_key(thrust::host,this->h_pp_pnt_idx,h_pp_pnt_idx+this->num_pp_pairs,this->h_pp_poly_idx);

        uint32_t * c_p_pnt_idx=new uint32_t[this->num_pp_pairs];
        uint32_t * c_p_pnt_len=new uint32_t[this->num_pp_pairs];
        assert(c_p_pnt_idx!=nullptr && c_p_pnt_len!=nullptr);

        uint32_t num_search_pnts=thrust::reduce_by_key(thrust::host,this->h_pp_pnt_idx,
            this->h_pp_pnt_idx+this->num_pp_pairs, thrust::constant_iterator<uint32_t>(1),
            c_p_pnt_idx,c_p_pnt_len).first-c_p_pnt_idx;
        std::cout<<"num_search_pnts="<<num_search_pnts<<std::endl;

        std::vector<uint32_t> c_pnt_idx_vec(c_p_pnt_idx,c_p_pnt_idx+num_search_pnts); 
        std::vector<uint32_t> c_pnt_len_vec(c_p_pnt_len,c_p_pnt_len+num_search_pnts);

        delete [] c_p_pnt_idx;
        delete [] c_p_pnt_len;

        std::vector<uint32_t> h_pnt_idx_vec;
        std::vector<uint32_t> h_pnt_len_vec;
        std::vector<uint32_t> h_poly_idx_vec;

        for(uint32_t k=0;k<num_pnts;k++)
        {
            OGRPoint pnt(h_pnt_x[k],h_pnt_y[k]);
            std::vector<uint32_t> temp_vec;
            for(uint32_t j=0;j<this->h_polygon_vec.size();j++)
            {
                if(this->h_polygon_vec[j]->Contains(&pnt))
                    temp_vec.push_back(j);
            }
            if(temp_vec.size()>0)
            {
                h_pnt_len_vec.push_back(temp_vec.size());
                h_pnt_idx_vec.push_back(k);
                h_poly_idx_vec.insert(h_poly_idx_vec.end(),temp_vec.begin(),temp_vec.end());
            }
        }
        CUDF_EXPECTS(c_pnt_idx_vec==h_pnt_idx_vec,"resulting point indices must be the same");
        CUDF_EXPECTS(c_pnt_len_vec==h_pnt_len_vec,"resulting numbers of polygons must be the same");

        uint32_t c_p=0,h_p=0;
        for(uint32_t k=0;k<h_pnt_idx_vec.size();k++)
        {
            EXPECT_EQ(c_pnt_idx_vec[k],h_pnt_idx_vec[k]);
            EXPECT_EQ(c_pnt_len_vec[k],h_pnt_len_vec[k]);
            std::vector<uint32_t> h_vec(h_poly_idx_vec.begin()+h_p,h_poly_idx_vec.begin()+h_p+h_pnt_len_vec[k]);
            std::vector<uint32_t> c_vec(h_pp_poly_idx+c_p,h_pp_poly_idx+c_p+c_pnt_len_vec[k]);
            CUDF_EXPECTS(h_vec==c_vec,"each polygon idx vec must be the same"); 
            h_p+=h_pnt_len_vec[k];
            c_p+=c_pnt_len_vec[k];
        }
        std::cout<<"pip_refine_test_large: verified"<<std::endl;
    }

    void tear_down()
    {
        delete[] h_pnt_x; h_pnt_x=nullptr;
        delete[] h_pnt_y; h_pnt_y=nullptr;

        delete[] h_pp_pnt_idx; h_pp_pnt_idx=nullptr;
        delete[] h_pp_poly_idx; h_pp_poly_idx=nullptr;
    }

};

TEST_F(PIPRefineTestLarge, test)
{
    const uint32_t num_levels=3;
    const uint32_t min_size=400;
    double scale=1.0;
    double x1=0,x2=8,y1=0,y2=8;

    this->setup_polygons();
    this->setup_points(x1,y1,x2,y2,scale,num_levels,min_size);
    
    std::cout<<"running test_point_large..........."<<std::endl;
    this->run_test(x1,y1,x2,y2,scale,num_levels,min_size);
    
    std::cout<<"verifying CPU and GPU results..........."<<std::endl;
    this->run_verify();
    std::cout<<"Verified"<< std::endl;
    
    this->tear_down();
}

