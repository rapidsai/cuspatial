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

//==========================================================================================================
//to be incorparated into io module

struct rec_cnyc
{
    int xf,yf,xt,yt;
};

  void VertexFromLinearRing(OGRLinearRing const& poRing, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        int nCount = poRing.getNumPoints();
        int nNewCount = aPointX.size() + nCount;
 
        aPointX.reserve( nNewCount );
        aPointY.reserve( nNewCount );
        for (int i = nCount - 1; i >= 0; i-- )
        {
            aPointX.push_back( poRing.getX(i));
            aPointY.push_back( poRing.getY(i));
        }
        aPartSize.push_back( nCount );
    }
 
     /*
      * Read a Polygon (could be with multiple rings) into x/y/size vectors
     */

    void LinearRingFromPolygon(OGRPolygon const & poPolygon, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        
        VertexFromLinearRing( *(poPolygon.getExteriorRing()),
                                        aPointX, aPointY, aPartSize );

        for(int i = 0; i < poPolygon.getNumInteriorRings(); i++ )
            VertexFromLinearRing( *(poPolygon.getInteriorRing(i)),
                                            aPointX, aPointY, aPartSize );
    }

     /*
      * Read a Geometry (could be MultiPolygon/GeometryCollection) into x/y/size vectors
     */

    void PolygonFromGeometry(OGRGeometry const *poShape, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        OGRwkbGeometryType eFlatType = wkbFlatten(poShape->getGeometryType());

        if (eFlatType == wkbMultiPolygon || eFlatType == wkbGeometryCollection )             
        {      
            OGRGeometryCollection *poGC = (OGRGeometryCollection *) poShape;
            for(int i = 0; i < poGC->getNumGeometries(); i++ )
            {
                OGRGeometry *poGeom=poGC->getGeometryRef(i);
                PolygonFromGeometry(poGeom,aPointX, aPointY, aPartSize );
            }
        }
        else if (eFlatType == wkbPolygon)
            LinearRingFromPolygon(*((OGRPolygon *) poShape),aPointX, aPointY, aPartSize );
        else
        {
           printf("error: must be polygonal geometry.\n" );
           exit(-1);
        }
    }

    int ReadLayer(const OGRLayerH layer,std::vector<int>& g_len_v,std::vector<int>&f_len_v,
         std::vector<int>& r_len_v,std::vector<double>& x_v, std::vector<double>& y_v)         
    {
        int num_feature=0;
        OGR_L_ResetReading(layer );
        OGRFeatureH hFeat;
        while( (hFeat = OGR_L_GetNextFeature( layer )) != NULL )
        {
            OGRGeometry *poShape=(OGRGeometry *)OGR_F_GetGeometryRef( hFeat );
            if(poShape==NULL)
            {
            	printf("Invalid Shape\n");
            	exit(-1);
            }
            
            std::vector<double> aPointX;
            std::vector<double> aPointY;
            std::vector<int> aPartSize;
            PolygonFromGeometry( poShape, aPointX, aPointY, aPartSize );

            x_v.insert(x_v.end(),aPointX.begin(),aPointX.end());
            y_v.insert(y_v.end(),aPointY.begin(),aPointY.end());
            r_len_v.insert(r_len_v.end(),aPartSize.begin(),aPartSize.end());
            f_len_v.push_back(aPartSize.size());
            OGR_F_Destroy( hFeat );
            num_feature++;
        }
        g_len_v.push_back(num_feature);
        return num_feature;
    }

size_t read_point_binary(const char *fn,double*& h_pnt_x,double*& h_pnt_y)
{
    FILE *fp=NULL;
    if((fp=fopen(fn,"rb"))==NULL)
    {
        printf("can not open %s\n",fn);
        exit(-1);
    }
    fseek (fp , 0 , SEEK_END);
    size_t sz=ftell (fp);
    assert(sz%sizeof(rec_cnyc)==0);
    size_t num_rec = sz/sizeof(rec_cnyc);
    printf("num_rec=%zd\n",num_rec);    
    fseek (fp , 0 , SEEK_SET);
    
    h_pnt_x=new double[num_rec];
    h_pnt_y=new double[num_rec];
    assert(h_pnt_x!=NULL && h_pnt_y!=NULL);
    struct rec_cnyc *temp=new rec_cnyc[num_rec];
 
    size_t t=fread(temp,sizeof(rec_cnyc),num_rec,fp);
    if(t!=num_rec)
    {
        printf("cny coord read error .....%10zd %10zd\n",t,num_rec);
        exit(-1);
    }
    for(uint32_t i=0;i<num_rec;i++)
    {
    	h_pnt_x[i]=temp[i].xf;
    	h_pnt_y[i]=temp[i].yf;
    } 	
    fclose(fp);
    delete[] temp;
    return num_rec;
    //timeval t0,t1;
    //gettimeofday(&t0, NULL);
}
//==========================================================================================================


struct SpatialJoinNYCTaxi : public GdfTest 
{    
    uint32_t num_pnt=0;
    uint32_t * d_pnt_id=NULL;
    double *d_h_pnt_x=NULL,*d_h_pnt_y=NULL;
    std::unique_ptr<cudf::column> col_pnt_id,col_pnt_x,col_pnt_y;
    
    uint32_t num_poly=0,num_ring=0,num_vertex=0;
    uint32_t *d_poly_id=NULL,*d_poly_fpos=NULL,*d_poly_rpos=NULL;
    double *d_poly_x=NULL,*d_poly_y=NULL;
    
    std::unique_ptr<cudf::column> col_poly_fpos,col_poly_rpos,col_poly_x,col_poly_y;
    
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    void setup_polygons(double& x1,double& y1,double& x2,double& y2)
    {
        const char* env_p = std::getenv("CUSPATIAL_DATA");
        CUDF_EXPECTS(env_p!=NULL,"CUSPATIAL_DATA environmental variable must be set");
        //from https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip
        std::string shape_filename=std::string(env_p)+std::string("taxi_zones.shp"); 
        std::cout<<"Using shapefile "<<shape_filename<<std::endl;

        std::vector<int> g_len_v,f_len_v,r_len_v;
        std::vector<double> x_v, y_v;
        GDALAllRegister();
        const char *file_name=shape_filename.c_str();
        GDALDatasetH hDS = GDALOpenEx(file_name, GDAL_OF_VECTOR, NULL, NULL, NULL );
        if(hDS==NULL)
        {
	    printf("Failed to open ESRI Shapefile dataset %s\n",file_name);
	    exit(-1);
        }
        OGRLayerH hLayer = GDALDatasetGetLayer( hDS,0 );
        int num_f=ReadLayer(hLayer,g_len_v,f_len_v,r_len_v,x_v,y_v);
        assert(num_f>0);

        uint32_t num_group=g_len_v.size();
        uint32_t num_poly=f_len_v.size();
        uint32_t num_ring=r_len_v.size();
        uint32_t num_vertex=x_v.size();
        //uint32_t num_temp=std::accumulate(r_len_v.begin(), r_len_v.end(),0);
        //printf("num_temp=%d num_vertex=%d\n",num_temp,num_vertex);

        uint32_t *h_poly_fpos=new uint32_t[num_poly];
        uint32_t *h_poly_rpos=new uint32_t[num_ring];
        double *h_poly_x=new double [num_vertex];
        double *h_poly_y=new double [num_vertex];

        std::copy_n(f_len_v.begin(),num_poly,h_poly_fpos);
        std::copy_n(r_len_v.begin(),num_ring,h_poly_rpos);
        std::copy_n(x_v.begin(),num_vertex,h_poly_x);
        std::copy_n(y_v.begin(),num_vertex,h_poly_y);
        printf("num_poly=%d num_ring=%d num_vertex=%d\n",num_poly,num_ring,num_vertex);

        x1=*(std::min_element(x_v.begin(),x_v.end()));
        x2=*(std::max_element(x_v.begin(),x_v.end()));
        y1=*(std::min_element(y_v.begin(),y_v.end()));
        y2=*(std::max_element(y_v.begin(),y_v.end()));
        printf("read_polygon_shape: x_min=%10.5f y_min=%10.5f, x_max=%10.5f, y_max=%10.5f\n",x1,y1, x2,y2);

        col_poly_fpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    		num_poly, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_fpos=cudf::mutable_column_device_view::create(col_poly_fpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_fpos!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_fpos, h_poly_fpos, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

        col_poly_rpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    		num_ring, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_rpos=cudf::mutable_column_device_view::create(col_poly_rpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_rpos!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_rpos, h_poly_rpos, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

        col_poly_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    		num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_x=cudf::mutable_column_device_view::create(col_poly_x->mutable_view(), stream)->data<double>();
        assert(d_poly_x!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_x, h_poly_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 

        col_poly_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    		num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_y=cudf::mutable_column_device_view::create(col_poly_y->mutable_view(), stream)->data<double>();
        assert(d_poly_y!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_y, h_poly_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );   

        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_fpos, h_poly_fpos, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) );    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_rpos, h_poly_rpos, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) );     
        thrust::inclusive_scan(thrust::device,d_poly_fpos,d_poly_fpos+num_poly,d_poly_fpos);
        thrust::inclusive_scan(thrust::device,d_poly_rpos,d_poly_rpos+num_ring,d_poly_rpos);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_x, h_poly_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_y, h_poly_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );     	

        delete[] h_poly_fpos;h_poly_fpos=NULL;
        delete[] h_poly_rpos;h_poly_rpos=NULL;
        delete[] h_poly_x; h_poly_x=NULL;
        delete[] h_poly_y; h_poly_y=NULL;   
    }
  
    void setup_points(double& x1,double& y1,double& x2,double& y2, uint32_t first_n)
    {
        const char* env_p = std::getenv("CUSPATIAL_DATA");
        CUDF_EXPECTS(env_p!=NULL,"CUSPATIAL_DATA environmental variable must be set");
        
        //from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page; 
        //pickup/drop-off locations are extracted and the lon/lat coordiates are converted to EPSG 2263 projection  
        std::string catalog_filename=std::string(env_p)+std::string("2009.cat"); 
        std::cout<<"Using catalog file "<<catalog_filename<<std::endl;
        
        std::vector<uint32_t> len_vec;
        std::vector<double *> x_vec;
        std::vector<double *> y_vec;
        uint32_t num=0;
        const char *file_name=catalog_filename.c_str();
        FILE *fp=NULL;
        if((fp=fopen(file_name,"r"))==NULL)
        {
   	    printf("Failed to open point catalog file %s\n",file_name);
	    exit(-2);      	
        }
        while(!feof(fp))
        {
             char str[500];
             int n1=fscanf(fp,"%s",str);
             printf("processing point data file %s\n",str);
             double *tmp_x=NULL,*tmp_y=NULL;
             size_t temp_len=read_point_binary(str,tmp_x,tmp_y);
             assert(tmp_x!=NULL && tmp_y!=NULL);
             num++;
             len_vec.push_back(temp_len);
             x_vec.push_back(tmp_x);
             y_vec.push_back(tmp_y);
             if(first_n>0 && num>=first_n) break;
        }    
        fclose(fp);

        for(uint32_t i=0;i<num;i++)
    	    num_pnt+=len_vec[i];
    
        uint32_t p=0;
        double *h_pnt_x=new double[num_pnt];
        double *h_pnt_y=new double[num_pnt];
        assert(h_pnt_x!=NULL && h_pnt_y!=NULL);
        for(uint32_t i=0;i<num;i++)
        {
            double *tmp_x=x_vec[i];
    	    double *tmp_y=y_vec[i];
    	    assert(tmp_x!=NULL && tmp_y!=NULL);
    	    int len=len_vec[i];
    	    std::copy(tmp_x,tmp_x+len,h_pnt_x+p);
    	    std::copy(tmp_y,tmp_y+len,h_pnt_y+p);
    	    p+=len;
    	    delete[] tmp_x;
    	    delete[] tmp_y;
        }
        assert(p==num_pnt);

        x1=*(std::min_element(h_pnt_x,h_pnt_x+num_pnt));
        x2=*(std::max_element(h_pnt_x,h_pnt_x+num_pnt));
        y1=*(std::min_element(h_pnt_y,h_pnt_y+num_pnt));
        y2=*(std::max_element(h_pnt_y,h_pnt_y+num_pnt));
        printf("read_point_catalog: x_min=%10.5f y_min=%10.5f, x_max=%10.5f, y_max=%10.5f\n",x1,y1, x2,y2);
        
        col_pnt_id = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
         	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_pnt_id=cudf::mutable_column_device_view::create(col_pnt_id->mutable_view(), stream)->data<uint32_t>();
        assert(d_pnt_id!=NULL);
        thrust::sequence(thrust::device,d_pnt_id,d_pnt_id+num_pnt);
      
        col_pnt_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
        	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_h_pnt_x=cudf::mutable_column_device_view::create(col_pnt_x->mutable_view(), stream)->data<double>();
        assert(d_h_pnt_x!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_h_pnt_x, h_pnt_x, num_pnt * sizeof(double), cudaMemcpyHostToDevice ) );    
    
        col_pnt_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
        	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_h_pnt_y=cudf::mutable_column_device_view::create(col_pnt_y->mutable_view(), stream)->data<double>();
        assert(d_h_pnt_y!=NULL);    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_h_pnt_y, h_pnt_y, num_pnt * sizeof(double), cudaMemcpyHostToDevice ) );    
    
    } 
 
    void run_test(double x1,double y1,double x2,double y2,double scale,uint32_t num_level,uint32_t min_size)
    {       
        cudf::mutable_column_view pnt_id_view=col_pnt_id->mutable_view();
        cudf::mutable_column_view pnt_x_view=col_pnt_x->mutable_view();
        cudf::mutable_column_view pnt_y_view=col_pnt_y->mutable_view();
        std::cout<<"run_test::num_pnt_view="<<pnt_id_view.size()<<std::endl;
        std::cout<<"run_test::num_pnt="<<col_pnt_id->size()<<std::endl;

        std::unique_ptr<cudf::experimental::table> quadtree= 
      		cuspatial::quadtree_on_points(pnt_id_view,pnt_x_view,pnt_y_view,x1,y1,x2,y2, scale,num_level, min_size);
        std::cout<<"run_test: quadtree num cols="<<quadtree->view().num_columns()<<std::endl;
     
        std::unique_ptr<cudf::experimental::table> bbox_tbl=
     	cuspatial::polygon_bbox(col_poly_fpos->view(),col_poly_rpos->view(),col_poly_x->view(),col_poly_y->view()); 
        std::cout<<"polygon bbox="<<bbox_tbl->view().num_rows()<<std::endl;
     
        const cudf::table_view quad_view=quadtree->view();
        const cudf::table_view bbox_view=bbox_tbl->view();
     
        std::unique_ptr<cudf::experimental::table> pq_pair_tbl=cuspatial::quad_bbox_join(
         quad_view,bbox_view,x1,y1,x2,y2, scale,num_level, min_size);   
        std::cout<<"polygon/quad num pair="<<pq_pair_tbl->view().num_columns()<<std::endl;
 
        const cudf::table_view pq_pair_view=pq_pair_tbl->view();
        const cudf::table_view pnt_view({pnt_id_view,pnt_x_view,pnt_y_view});
 
        std::unique_ptr<cudf::experimental::table> pip_pair_tbl=cuspatial::pip_refine(
       	  pq_pair_view,quad_view,pnt_view,
         col_poly_fpos->view(),col_poly_rpos->view(),col_poly_x->view(),col_poly_y->view());   
        std::cout<<"polygon/point num pair="<<pip_pair_tbl->view().num_columns()<<std::endl;
    }
 
};

TEST_F(SpatialJoinNYCTaxi, test)
{
    const uint32_t num_level=15;
    const uint32_t min_size=512;
    const uint32_t first_n=1;
  
    double poly_x1,poly_y1,poly_x2,poly_y2;
    this->setup_polygons(poly_x1,poly_y1,poly_x2,poly_y2);
        
    double pnt_x1,pnt_y1,pnt_x2,pnt_y2;
    this->setup_points(pnt_x1,pnt_y1,pnt_x2,pnt_y2,first_n);
 
    double width=poly_x2-poly_x1;
    double height=poly_y2-poly_y1;
    double length=(width>height)?width:height;
    double scale=length/((1<<num_level)+2);
    double bbox_x1=poly_x1-scale; 
    double bbox_y1=poly_y1-scale;
    double bbox_x2=poly_x2+scale; 
    double bbox_y2=poly_y2+scale;
    printf("length=%15.10f scale=%15.10f\n",length,scale);

    std::cout<<"running test_point_large..........."<<std::endl;
    this->run_test(bbox_x1,bbox_y1,bbox_x2,bbox_y2,scale,num_level,min_size);
}

