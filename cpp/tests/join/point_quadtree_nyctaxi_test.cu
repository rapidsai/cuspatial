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

#include <ogrsf_frmts.h>
#include <geos_c.h>


#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

#include <utility/utility.hpp>
#include <utility/helper_thrust.cuh>
#include <utility/quadtree_thrust.cuh>
#include <utility/bbox_thrust.cuh>

#include <cuspatial/quadtree.hpp>
#include <cuspatial/polygon_bbox.hpp>
#include <cuspatial/spatial_jion.hpp>

#include "spatial_join_test_utility.hpp"

struct SpatialJoinNYCTaxiTest : public GdfTest 
{        
    uint32_t num_pnts=0;

    uint32_t num_quadrants=0;

    uint32_t num_pq_pairs=0;

    uint32_t num_pp_pairs=0;
   
    //point x/y on device, shared between setup_points and run_test
    //the life span of d_pnt_x/d_pnt_y depends on col_pnt_x/col_pnt_y
    double *d_pnt_x=nullptr,*d_pnt_y=nullptr;

    //point x/y on host
    double *h_pnt_x=nullptr,*h_pnt_y=nullptr;

    uint32_t num_poly=0,num_ring=0,num_vertex=0;

    //polygon vertices x/y
    double *h_poly_x=nullptr,*h_poly_y=nullptr;

    //quadtree length/fpos
    uint32_t *h_qt_length=nullptr,*h_qt_fpos=nullptr;   

    //quadrant/polygon pairs
    uint32_t *h_pq_quad_idx=nullptr,*h_pq_poly_idx=nullptr;   
    
    //point/polygon pairs on device; shared between run_test and compute_mismatch
    //the life span of d_pp_pnt_idx/d_pp_poly_idx depends on pip_pair_tbl
    uint32_t *h_pp_pnt_idx=nullptr,*h_pp_poly_idx=nullptr;

    //poygons using GDAL/OGR OGRGeometry structure
    std::vector<OGRGeometry *> h_ogr_polygon_vec;

    //sequential idx 0..num_poly-1 to index h_ogr_polygon_vec
    //needed when actual polygons in spatial join are only a subset, e.g., multi-polygons only  
    std::vector<uint32_t> h_org_poly_idx_vec;
    
    std::unique_ptr<cudf::column> col_pnt_x,col_pnt_y;

    std::unique_ptr<cudf::column> col_poly_fpos,col_poly_rpos,col_poly_x,col_poly_y;    

    //memory allocated for these structures could be released dynamically, should this be an issue
    std::unique_ptr<cudf::experimental::table> quadtree_tbl,bbox_tbl,pq_pair_tbl,pip_pair_tbl;

    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    SBBox<double> setup_polygons(const char *file_name,uint8_t type)
    {
        std::vector<int> g_len_v,f_len_v,r_len_v;
        std::vector<double> x_v, y_v;
        GDALAllRegister();
        GDALDatasetH hDS = GDALOpenEx(file_name, GDAL_OF_VECTOR, nullptr, nullptr, nullptr );
        if(hDS==nullptr)
        {
            std::cout<<"Failed to open ESRI Shapefile dataset "<< file_name<<std::endl;
            exit(-1);
        }
        //a shapefile abstracted as a GDALDatasetGetLayer typically has only one layer
        OGRLayerH hLayer = GDALDatasetGetLayer( hDS,0 );

        this->h_ogr_polygon_vec.clear();
        this->h_org_poly_idx_vec.clear();
        
        //type: 0 for all, 1 for simple polygons and 2 for multi-polygons
        uint32_t num_f=ReadLayer(hLayer,g_len_v,f_len_v,r_len_v,x_v,y_v,type,h_ogr_polygon_vec,h_org_poly_idx_vec);
        assert(num_f>0);
        
        //num_group=g_len_v.size();
        this->num_poly=f_len_v.size();
        this->num_ring=r_len_v.size();
        this->num_vertex=x_v.size();

        uint32_t *h_poly_flen=new uint32_t[num_poly];
        uint32_t *h_poly_rlen=new uint32_t[num_ring];
        assert(h_poly_flen!=nullptr && h_poly_rlen!=nullptr);
        
        this->h_poly_x=new double [num_vertex];
        this->h_poly_y=new double [num_vertex];
        assert(h_poly_x!=nullptr && h_poly_y!=nullptr);

        std::copy_n(f_len_v.begin(),num_poly,h_poly_flen);
        std::copy_n(r_len_v.begin(),num_ring,h_poly_rlen);
        std::copy_n(x_v.begin(),num_vertex,h_poly_x);
        std::copy_n(y_v.begin(),num_vertex,h_poly_y);
        std::cout<<"setup_polygons: num_poly="<<num_poly<<" num_ring="<<num_ring<<" num_vertex="<<num_vertex<<std::endl;

        //note that the bbox of all polygons will used as the Area of Intersects (AOI) to join points with polygons 
        double x1=*(std::min_element(x_v.begin(),x_v.end()));
        double x2=*(std::max_element(x_v.begin(),x_v.end()));
        double y1=*(std::min_element(y_v.begin(),y_v.end()));
        double y2=*(std::max_element(y_v.begin(),y_v.end()));
        std::cout<<"read_polygon_bbox: x_min="<<x1<<"  y_min="<<y1<<" x_max="<<x2<<" y_max="<<y2<<std::endl;

        //create columns for polygons and populate their data arrays from host values
        col_poly_fpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32},
            num_poly, cudf::mask_state::UNALLOCATED, stream, mr );      
        uint32_t *d_poly_fpos=cudf::mutable_column_device_view::create(col_poly_fpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_fpos!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_fpos, h_poly_flen, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

        this->col_poly_rpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32},
            num_ring, cudf::mask_state::UNALLOCATED, stream, mr );      
        uint32_t *d_poly_rpos=cudf::mutable_column_device_view::create(col_poly_rpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_rpos!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_rpos, h_poly_rlen, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) );

        this->col_poly_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64},
            num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
        double *d_poly_x=cudf::mutable_column_device_view::create(col_poly_x->mutable_view(), stream)->data<double>();
        assert(d_poly_x!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_x, h_poly_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );

        this->col_poly_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64},
            num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
        double *d_poly_y=cudf::mutable_column_device_view::create(col_poly_y->mutable_view(), stream)->data<double>();
        assert(d_poly_y!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_y, h_poly_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );

        //copy ring/vertex lengths from CPU to GPU
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_fpos, h_poly_flen, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) );
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_rpos, h_poly_rlen, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) );
        
        delete[] h_poly_flen;h_poly_flen=nullptr;
        delete[] h_poly_rlen;h_poly_rlen=nullptr;

        //in-place scan (prefix-sum) to accumulate lengths to offsets 
        thrust::inclusive_scan(thrust::device,d_poly_fpos,d_poly_fpos+num_poly,d_poly_fpos);
        thrust::inclusive_scan(thrust::device,d_poly_rpos,d_poly_rpos+num_ring,d_poly_rpos);

        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_x, h_poly_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_y, h_poly_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );

        return SBBox<double>(thrust::make_tuple(x1,y1), thrust::make_tuple(x2,y2));
    }

    SBBox<double> setup_points(const char * file_name, uint32_t first_n)
    {

        //read invidual data file  
        std::vector<uint32_t> len_vec;
        std::vector<double *> x_vec;
        std::vector<double *> y_vec;
        uint32_t num=0;
        
        FILE *fp=nullptr;
        if((fp=fopen(file_name,"r"))==nullptr)
        {
           std::cout<<"Failed to open point catalog file "<<file_name<<std::endl;
           exit(-2);          
        }
        while(!feof(fp))
        {
             char str[500];
             int n1=fscanf(fp,"%s",str);
             std::cout<<"processing point data file "<<str<<std::endl;
             double *tmp_x=nullptr,*tmp_y=nullptr;
             size_t temp_len=read_point_binary(str,tmp_x,tmp_y);
             assert(tmp_x!=nullptr && tmp_y!=nullptr);
             num++;
             len_vec.push_back(temp_len);
             x_vec.push_back(tmp_x);
             y_vec.push_back(tmp_y);
             if(first_n>0 && num>=first_n) break;
        }    
        fclose(fp);

        //prepare memory allocation
        for(uint32_t i=0;i<num;i++)
            num_pnts+=len_vec[i];
        uint32_t p=0;
        this->h_pnt_x=new double[num_pnts];
        this->h_pnt_y=new double[num_pnts];
        assert(h_pnt_x!=nullptr && h_pnt_y!=nullptr);
        
        //concatination
        for(uint32_t i=0;i<num;i++)
        {
            double *tmp_x=x_vec[i];
            double *tmp_y=y_vec[i];
            assert(tmp_x!=nullptr && tmp_y!=nullptr);
            int len=len_vec[i];
            std::copy(tmp_x,tmp_x+len,h_pnt_x+p);
            std::copy(tmp_y,tmp_y+len,h_pnt_y+p);
            p+=len;
            delete[] tmp_x;
            delete[] tmp_y;
        }
        assert(p==num_pnts);

        //compute the bbox of all points; outlier points may have irrational values
        //any points that do not fall within the Area of Interests (AOIs) will be assgin a special Morton code
        //AOI is user-defined and is passed to quadtree indexing and spatial join 
        double x1=*(std::min_element(h_pnt_x,h_pnt_x+num_pnts));
        double x2=*(std::max_element(h_pnt_x,h_pnt_x+num_pnts));
        double y1=*(std::min_element(h_pnt_y,h_pnt_y+num_pnts));
        double y2=*(std::max_element(h_pnt_y,h_pnt_y+num_pnts));
        std::cout<<"read_point_catalog: x_min="<<x1<<"  y_min="<<y1<<" x_max="<<x2<<" y_max="<<y2<<std::endl;

        //create x/y columns, expose their raw pointers to be used in run_test() and populate x/y arrays
        this->col_pnt_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            num_pnts, cudf::mask_state::UNALLOCATED, stream, mr );      
        this->d_pnt_x=cudf::mutable_column_device_view::create(col_pnt_x->mutable_view(), stream)->data<double>();
        assert(this->d_pnt_x!=nullptr);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_x, h_pnt_x, num_pnts * sizeof(double), cudaMemcpyHostToDevice ) );

        this->col_pnt_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
            num_pnts, cudf::mask_state::UNALLOCATED, stream, mr );      
        this->d_pnt_y=cudf::mutable_column_device_view::create(col_pnt_y->mutable_view(), stream)->data<double>();
        assert(this->d_pnt_y!=nullptr);    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_y, h_pnt_y, num_pnts * sizeof(double), cudaMemcpyHostToDevice ) );
        
        return SBBox<double>(thrust::make_tuple(x1,y1), thrust::make_tuple(x2,y2));
    }

    void run_test(double x1,double y1,double x2,double y2,double scale,uint32_t num_level,uint32_t min_size)
    {
        timeval t0,t1,t2,t3;

        gettimeofday(&t0, nullptr); 
        cudf::mutable_column_view pnt_x_view=col_pnt_x->mutable_view();
        cudf::mutable_column_view pnt_y_view=col_pnt_y->mutable_view();
        std::cout<<"run_test::num_pnts="<<col_pnt_x->size()<<std::endl;

        gettimeofday(&t2, nullptr);
        this->quadtree_tbl= cuspatial::quadtree_on_points(pnt_x_view,pnt_y_view,x1,y1,x2,y2, scale,num_level, min_size);
        this->num_quadrants=this->quadtree_tbl->view().num_rows();
        std::cout<<"# of quadrants="<<this->num_quadrants<<std::endl;
        gettimeofday(&t3, nullptr);
        float quadtree_time=cuspatial::calc_time("quadtree_tbl constrution time=",t2,t3);

   
    void tear_down()
    {
        delete[] this->h_poly_x; this->h_poly_x=nullptr;
        delete[] this->h_poly_y; this->h_poly_y=nullptr;

        delete[] this->h_pnt_x; this->h_pnt_x=nullptr;
        delete[] this->h_pnt_y; this->h_pnt_y=nullptr;
        
        delete[] this->h_pq_quad_idx; this->h_pq_quad_idx=nullptr;
        delete[] h_pq_poly_idx; h_pq_poly_idx=nullptr;
        
        delete[] this->h_qt_length; this->h_qt_length=nullptr;
        delete[] this->h_qt_fpos; this->h_qt_fpos=nullptr;
    }

};



/* 
 * There could be multple configureations (minior ones are inside parentheses): 
 * pick one of three polygon datasets
 * choose from compare_random_points and compare_matched_pairs 
 * (vary first_n from 1-12 to pick the points of the first n months) 
 * (set poly_type to 0,1,2 where 0 is the deafult)
*/

TEST_F(SpatialJoinNYCTaxiTest, test)
{
    const char* env_p = std::getenv("CUSPATIAL_DATA");
    CUDF_EXPECTS(env_p!=nullptr,"CUSPATIAL_DATA environmental variable must be set");
    
    const uint32_t num_level=15;
    const uint32_t min_size=512;
    const uint32_t first_n=1; 

    std::cout<<"loading NYC taxi pickup locations..........."<<std::endl;
    
    //from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page; 
    //pickup/drop-off locations are extracted and the lon/lat coordiates are converted to epsg:2263 projection
        
    //a catalog file is simply a collection of invidual binary data files with a pre-defined structure
    //each line repersents a data file, e.g., pickup+drop-off locations for a month
    std::string catalog_filename=std::string(env_p)+std::string("2009.cat"); 
    std::cout<<"Using catalog file "<<catalog_filename<<std::endl;
    this->setup_points(catalog_filename.c_str(),first_n);

    std::cout<<"loading NYC polygon data..........."<<std::endl;

    enum POLYID {taxizone_id=0,cd_id,ct_id};    
    POLYID sel_id=taxizone_id;

    const char * shape_files[]={"taxi_zones.shp","nycd_11a_av/nycd.shp","nyct2000_11a_av/nyct2000.shp"};
    
    const char * bin_files[]={"nyc_taxizone_2009_1.bin","nyc_cd_2009_12.bin","nyc_ct_2009_12.bin"};
 
    std::cout<<"loading NYC polygon data..........."<<std::endl;

    std::string shape_filename=std::string(env_p)+std::string(shape_files[sel_id]); 
    
    std::cout<<"Using shapefile "<<shape_filename<<std::endl;

    //uint8_t poly_type=2; //multi-polygons only 
    //uint8_t poly_type=1; //single-polygons only 
    uint8_t poly_type=0; //all polygons
    SBBox<double> aoi=this->setup_polygons(shape_filename.c_str(),poly_type);

    double poly_x1=thrust::get<0>(aoi.first);
    double poly_y1=thrust::get<1>(aoi.first);
    double poly_x2=thrust::get<0>(aoi.second);
    double poly_y2=thrust::get<1>(aoi.second);
    
    double width=poly_x2-poly_x1;
    double height=poly_y2-poly_y1;
    double length=(width>height)?width:height;
    double scale=length/((1<<num_level)+2);
    double bbox_x1=poly_x1-scale;
    double bbox_y1=poly_y1-scale;
    double bbox_x2=poly_x2+scale; 
    double bbox_y2=poly_y2+scale;
    printf("Area of Interests: length=%15.10f scale=%15.10f\n",length,scale);

    std::cout<<"running test on NYC taxi trip data..........."<<std::endl;
    this->run_test(bbox_x1,bbox_y1,bbox_x2,bbox_y2,scale,num_level,min_size);
    
    this->write_nyc_taxi(bin_files[sel_id]);

    this->tear_down();

}//TEST_F

